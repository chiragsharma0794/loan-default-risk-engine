"""Stage 5: fixed-configuration refit and one final holdout comparison."""
import json
from pathlib import Path
from time import perf_counter
from importlib.metadata import version

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

try:
    from .lgbm_model import PROJECT_ROOT, load_model_inputs, file_hash
    from .stratified_sampling import RAW_DATA_PATH, VALIDATION_PATH, read_contract, assign_splits, TARGET_MAP
    from .feature_screening import transform_features
except ImportError:
    from lgbm_model import PROJECT_ROOT, load_model_inputs, file_hash
    from stratified_sampling import RAW_DATA_PATH, VALIDATION_PATH, read_contract, assign_splits, TARGET_MAP
    from feature_screening import transform_features

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'final_model'
COMPARISON_PATH = PROJECT_ROOT / 'outputs' / 'model_comparison.csv'
THRESHOLD = 0.5
BOOTSTRAP_REPLICATES = 500
SEED = 42
SELECTION_RULE = 'Select tuned only if holdout AUC is higher and paired stratified-bootstrap 95% percentile interval for tuned-minus-baseline AUC has lower bound > 0; otherwise retain baseline.'


def evaluate_predictions(y, probabilities):
    y, probabilities = np.asarray(y, dtype=int), np.asarray(probabilities, dtype=float)
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError('Invalid probabilities')
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Both outcome classes are required')
    predicted = probabilities >= THRESHOLD
    tp = int(((y == 1) & predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    return {
        'roc_auc': float(roc_auc_score(y, probabilities)),
        'average_precision': float(average_precision_score(y, probabilities)),
        'log_loss': float(log_loss(y, probabilities, labels=[0, 1])),
        'brier_score': float(brier_score_loss(y, probabilities)),
        'threshold': THRESHOLD, 'threshold_rule': 'positive if probability >= 0.5',
        'precision': tp / (tp + fp) if tp + fp else None,
        'recall': tp / (tp + fn), 'predicted_positive_rows': tp + fp,
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'mean_probability': float(probabilities.mean()),
        'observed_default_rate': float(y.mean()),
        'mean_prediction_minus_default_rate': float(probabilities.mean() - y.mean()),
    }


def calibration_table(y, probabilities, model_name):
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    bins = np.minimum((probabilities * 10).astype(int), 9)
    rows = []
    for index in range(10):
        mask = bins == index
        # Fixed probability bins: left inclusive, right exclusive except 1.0.
        rows.append({'model': model_name, 'bin_lower': index / 10, 'bin_upper': (index + 1) / 10,
                     'rows': int(mask.sum()), 'defaults': int(y[mask].sum()),
                     'mean_probability': float(probabilities[mask].mean()) if mask.any() else None,
                     'observed_default_rate': float(y[mask].mean()) if mask.any() else None})
    return rows


def weighted_auc_from_bins(positive_weights, negative_weights):
    """Mann-Whitney probability with half credit for tied scores."""
    positive_weights, negative_weights = np.asarray(positive_weights), np.asarray(negative_weights)
    denominator = positive_weights.sum() * negative_weights.sum()
    if denominator <= 0:
        raise ValueError('Both classes need positive total weight')
    return float(np.sum(positive_weights * (np.cumsum(negative_weights) - 0.5 * negative_weights)) / denominator)


def paired_auc_interval(y, baseline_probabilities, tuned_probabilities, replicates=BOOTSTRAP_REPLICATES):
    """Paired, class-stratified loan-row bootstrap; sort scores once for efficiency."""
    y = np.asarray(y, dtype=int)
    positive, negative = y == 1, y == 0
    m, n = int(positive.sum()), int(negative.sum())
    if min(m, n) == 0:
        raise ValueError('Both classes are required for stratified bootstrap')
    layouts = []
    for probabilities in (baseline_probabilities, tuned_probabilities):
        values, inverse = np.unique(probabilities, return_inverse=True)
        layouts.append((inverse[positive], inverse[negative], len(values)))
    rng = np.random.default_rng(SEED)
    differences = np.empty(replicates)
    positive_chances, negative_chances = np.full(m, 1 / m), np.full(n, 1 / n)
    for index in range(replicates):
        positive_counts = rng.multinomial(m, positive_chances)
        negative_counts = rng.multinomial(n, negative_chances)
        aucs = []
        for positive_bins, negative_bins, size in layouts:
            pw = np.bincount(positive_bins, weights=positive_counts, minlength=size)
            nw = np.bincount(negative_bins, weights=negative_counts, minlength=size)
            aucs.append(weighted_auc_from_bins(pw, nw))
        differences[index] = aucs[1] - aucs[0]
    low, high = np.quantile(differences, [0.025, 0.975])
    return {'method': 'paired class-stratified percentile bootstrap of loan rows',
            'replicates': replicates, 'seed': SEED, 'confidence_level': 0.95,
            'lower': float(low), 'upper': float(high),
            'assumption': 'Rows independent; unavailable borrower IDs prevent borrower-cluster uncertainty estimates.'}


def load_holdout(contract):
    schema = contract['preprocessing']
    usecols = list(dict.fromkeys(schema['raw_features'] + ['issue_d', 'loan_status', 'term']))
    features, labels = [], []
    for chunk in pd.read_csv(RAW_DATA_PATH, usecols=usecols, dtype='string', chunksize=50000):
        selected = chunk.loc[assign_splits(chunk).eq('holdout')]
        if selected.empty:
            continue
        dates = pd.to_datetime(selected.issue_d, format='%b-%Y')
        assert dates.dt.year.eq(2014).all()
        features.append(transform_features(selected, schema))
        labels.append(selected.loan_status.map(TARGET_MAP).astype('int8'))
    x, y = pd.concat(features), pd.concat(labels)
    assert len(x) == contract['splits']['holdout']['rows'] and x.index.is_unique
    assert x.columns.tolist() == schema['features']
    return x, y


def run_evaluation(output_dir=OUTPUT_DIR, comparison_path=COMPARISON_PATH):
    output_dir, comparison_path = Path(output_dir), Path(comparison_path)
    if (output_dir.exists() and any(output_dir.iterdir())) or comparison_path.exists():
        raise FileExistsError('Final evaluation already exists or was started; do not silently rescore the holdout')
    started = perf_counter()
    baseline_dir, tuning_dir = PROJECT_ROOT / 'outputs/baseline', PROJECT_ROOT / 'outputs/tuning'
    baseline = json.loads((baseline_dir / 'metrics.json').read_text())
    tuning = json.loads((tuning_dir / 'best_params.json').read_text())
    frozen_hashes = {str(p.relative_to(PROJECT_ROOT)): file_hash(p)
                     for folder in (baseline_dir, tuning_dir) for p in folder.iterdir() if p.is_file()}
    for name, digest in tuning['baseline_artifact_sha256'].items():
        assert file_hash(baseline_dir / name) == digest
    for name, digest in tuning['artifact_sha256'].items():
        assert file_hash(tuning_dir / name) == digest
    for name, digest in baseline['code_sha256'].items():
        assert file_hash(PROJECT_ROOT / name) == digest
    assert file_hash(PROJECT_ROOT / 'src/tune_lgbm.py') == tuning['tuning_code_sha256']
    assert file_hash(VALIDATION_PATH) == baseline['preprocessing_contract']['sha256']
    for package, pinned in baseline['environment']['packages'].items():
        assert version(package) == pinned
    contract = read_contract()
    print('Verifying fixed candidate refit using training only, before opening holdout...', flush=True)
    x_train, x_validation, y_train, y_validation = load_model_inputs()
    baseline_model = lgb.Booster(model_file=str(baseline_dir / 'model.txt'))
    saved_candidate = lgb.Booster(model_file=str(tuning_dir / 'candidate_model.txt'))
    fixed_parameters = tuning['fixed_iteration_parameters_for_stage5']
    assert fixed_parameters['n_estimators'] == tuning['best_iteration'] == 57
    refit_started = perf_counter()
    candidate = lgb.LGBMClassifier(**fixed_parameters)
    candidate.fit(x_train, y_train)  # No eval_set, holdout or iteration selection here.
    refit_seconds = perf_counter() - refit_started
    candidate_model = candidate.booster_
    refit_validation = candidate_model.predict(x_validation)
    frozen_validation = saved_candidate.predict(x_validation)
    np.testing.assert_allclose(refit_validation, frozen_validation, rtol=1e-12, atol=1e-15)
    saved_rows = pd.read_csv(tuning_dir / 'validation_predictions.csv', float_precision='round_trip')
    np.testing.assert_array_equal(saved_rows.source_row, x_validation.index)
    np.testing.assert_array_equal(saved_rows.actual_default, y_validation)
    np.testing.assert_allclose(refit_validation, saved_rows.predicted_probability, rtol=1e-12, atol=1e-15)
    assert candidate_model.current_iteration() == tuning['best_iteration']
    assert baseline_model.current_iteration() == baseline['best_iteration']
    assert candidate_model.feature_name() == baseline_model.feature_name() == contract['preprocessing']['features']
    development_rows = x_train.index.union(x_validation.index)
    del x_train, y_train
    # Persist the predeclared decision rule and frozen model identities before holdout access.
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {'stage': 5, 'status': 'holdout_evaluation_started', 'selection_rule': SELECTION_RULE,
              'threshold': THRESHOLD, 'bootstrap_replicates': BOOTSTRAP_REPLICATES,
              'frozen_artifact_sha256': frozen_hashes, 'preprocessing_contract': baseline['preprocessing_contract'],
              'source_sha256': contract['source']['sha256'], 'policy': contract['policy'],
              'baseline_parameters': baseline['classifier_parameters'], 'baseline_iterations': baseline['best_iteration'],
              'tuned_fixed_parameters': fixed_parameters, 'tuned_iterations': tuning['best_iteration'],
              'training_rows': contract['splits']['train']['rows'], 'feature_count': len(contract['preprocessing']['features']),
              'pre_holdout_refit_validation_max_abs_difference': float(np.max(np.abs(refit_validation-frozen_validation))),
              'environment': baseline['environment'], 'evaluation_code_sha256': file_hash(Path(__file__))}
    (output_dir / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('Opening the frozen 2014 holdout; both configurations and selection rule are fixed.', flush=True)
    x_holdout, y_holdout = load_holdout(contract)
    assert x_holdout.index.intersection(development_rows).empty
    assert x_holdout.dtypes.equals(x_validation.dtypes)
    baseline_probabilities = baseline_model.predict(x_holdout)
    tuned_probabilities = candidate_model.predict(x_holdout)
    results = {'baseline': evaluate_predictions(y_holdout, baseline_probabilities),
               'tuned': evaluate_predictions(y_holdout, tuned_probabilities)}
    predictions = pd.DataFrame({'source_row': x_holdout.index, 'actual_default': y_holdout.to_numpy(),
                                'baseline_probability': baseline_probabilities, 'tuned_probability': tuned_probabilities})
    predictions.to_csv(output_dir / 'holdout_predictions.csv', index=False, float_format='%.17g')
    print('Both holdout predictions saved; computing paired uncertainty without further fitting...', flush=True)
    interval = paired_auc_interval(y_holdout, baseline_probabilities, tuned_probabilities)
    differences = {key: results['tuned'][key]-results['baseline'][key]
                   for key in ('roc_auc', 'average_precision', 'log_loss', 'brier_score')}
    selected = 'tuned' if differences['roc_auc'] > 0 and interval['lower'] > 0 else 'baseline'
    selected_model = candidate_model if selected == 'tuned' else baseline_model
    selected_model.save_model(str(output_dir / 'model.txt'))
    # Serialization verification uses validation, not a second holdout scoring pass.
    reloaded = lgb.Booster(model_file=str(output_dir / 'model.txt'))
    expected_validation = refit_validation if selected == 'tuned' else baseline_model.predict(x_validation)
    np.testing.assert_allclose(reloaded.predict(x_validation), expected_validation, rtol=1e-12, atol=1e-15)
    pd.DataFrame(calibration_table(y_holdout, baseline_probabilities, 'baseline') +
                 calibration_table(y_holdout, tuned_probabilities, 'tuned')).to_csv(
                     output_dir / 'calibration.csv', index=False, float_format='%.17g')
    comparison = pd.DataFrame([{'metric': key, 'baseline_holdout': results['baseline'][key],
                               'tuned_holdout': results['tuned'][key], 'tuned_minus_baseline': differences[key],
                               'better_direction': 'higher' if key in ('roc_auc','average_precision') else 'lower'}
                              for key in differences])
    comparison.to_csv(comparison_path, index=False, float_format='%.17g')
    for name, digest in frozen_hashes.items():
        assert file_hash(PROJECT_ROOT / name) == digest
    assert file_hash(VALIDATION_PATH) == baseline['preprocessing_contract']['sha256']
    report.update({'status': 'complete_holdout_consumed', 'selected_model': selected,
                   'selected_parameters': fixed_parameters if selected == 'tuned' else {**baseline['classifier_parameters'], 'n_estimators': baseline['best_iteration']},
                   'holdout': {'rows': len(y_holdout), 'positive_rows': int(y_holdout.sum()),
                               'default_rate': float(y_holdout.mean()), 'source_row_unique': True,
                               'development_overlap_rows': 0},
                   'holdout_metrics': results, 'tuned_minus_baseline': differences,
                   'paired_auc_difference_interval': interval,
                   'generalization_conclusion': ('Tuned AUC improvement supported on this holdout under the row-independence bootstrap assumption.' if selected == 'tuned' else 'Tuned AUC improvement not supported sufficiently by the predeclared rule; retain baseline.'),
                   'calibration_method': '10 fixed equal-width probability bins; [left,right) except last bin includes 1.0',
                   'verification': {'frozen_inputs_unchanged': True, 'saved_model_validation_predictions_match': True,
                                    'candidate_refit_matches_frozen_validation_predictions': True},
                   'runtime_seconds': {'fixed_candidate_refit': refit_seconds, 'total_through_checks': perf_counter()-started},
                   'limitations': baseline['limitations'][:3] + ['Holdout is now consumed; do not retune or reselect using this set.',
                                      'Row-bootstrap uncertainty does not account for unidentified repeat borrowers.',
                                      '0.5 is a descriptive threshold, not a cost-optimized lending decision.'],
                   'artifact_sha256': {name: file_hash(output_dir/name) for name in
                                       ['model.txt','holdout_predictions.csv','calibration.csv']},
                   'comparison_sha256': file_hash(comparison_path)})
    (output_dir / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['selected_model','holdout','holdout_metrics',
                      'tuned_minus_baseline','paired_auc_difference_interval','runtime_seconds']},indent=2),flush=True)
    return report


if __name__ == '__main__':
    run_evaluation()
