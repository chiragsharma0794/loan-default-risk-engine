"""Stage 4: fixed-budget validation-only random search; never loads a holdout matrix."""
import json
from pathlib import Path
from time import perf_counter
from importlib.metadata import version

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

try:
    from .lgbm_model import load_model_inputs, file_hash, PROJECT_ROOT
except ImportError:
    from lgbm_model import load_model_inputs, file_hash, PROJECT_ROOT

TRIAL_COUNT = 24
SEARCH_SEED = 42
BASELINE_DIR = PROJECT_ROOT / 'outputs' / 'baseline'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'tuning'
SEARCH_SPACE = {
    'learning_rate': {'distribution': 'log-uniform', 'low': 0.015, 'high': 0.12},
    'num_leaves': [7, 15, 31, 63],
    'max_depth': [3, 5, 7, 9, -1],
    'num_leaves_constraint': 'cap drawn num_leaves at 2**max_depth when max_depth > 0',
    'min_child_samples': [20, 50, 100, 200, 400],
    'colsample_bytree': [0.7, 0.85, 1.0],
    'subsample': [0.7, 0.85, 1.0],
    'subsample_freq': '1 if subsample < 1, else 0',
    'reg_alpha': {'distribution': 'log-uniform', 'low': 0.0001, 'high': 10.0},
    'reg_lambda': {'distribution': 'log-uniform', 'low': 0.001, 'high': 30.0},
    'min_split_gain': [0.0, 0.01, 0.05, 0.1],
    'n_estimators': 1500,
}


def sample_candidates():
    rng = np.random.default_rng(SEARCH_SEED)
    candidates = []
    for _ in range(TRIAL_COUNT):
        depth = int(rng.choice(SEARCH_SPACE['max_depth']))
        leaves = int(rng.choice(SEARCH_SPACE['num_leaves']))
        fraction = float(rng.choice(SEARCH_SPACE['subsample']))
        params = {
            'max_depth': depth,
            'num_leaves': min(leaves, 2 ** depth) if depth > 0 else leaves,
            'subsample': fraction,
            'subsample_freq': 1 if fraction < 1 else 0,
            'min_child_samples': int(rng.choice(SEARCH_SPACE['min_child_samples'])),
            'colsample_bytree': float(rng.choice(SEARCH_SPACE['colsample_bytree'])),
            'min_split_gain': float(rng.choice(SEARCH_SPACE['min_split_gain'])),
            'n_estimators': SEARCH_SPACE['n_estimators'],
        }
        for name in ('learning_rate', 'reg_alpha', 'reg_lambda'):
            spec = SEARCH_SPACE[name]
            params[name] = float(np.exp(rng.uniform(np.log(spec['low']), np.log(spec['high']))))
        candidates.append(params)
    return candidates


def score(y, probabilities):
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError('Invalid candidate probabilities')
    return {
        'roc_auc': float(roc_auc_score(y, probabilities)),
        'average_precision': float(average_precision_score(y, probabilities)),
        'log_loss': float(log_loss(y, probabilities, labels=[0, 1])),
        'brier_score': float(brier_score_loss(y, probabilities)),
    }


def save_candidate_model(booster, path, validation_features, expected_probabilities):
    # Native serialization preserves LightGBM's byte-counted tree offsets on Windows.
    booster.save_model(str(path))
    loaded = lgb.Booster(model_file=str(path))
    actual = loaded.predict(validation_features)
    np.testing.assert_allclose(actual, expected_probabilities, rtol=1e-12, atol=1e-15)
    assert loaded.current_iteration() == booster.current_iteration()
    return float(np.max(np.abs(actual - expected_probabilities)))


def run_tuning(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Tuning output is frozen; do not overwrite or silently extend the search')
    started = perf_counter()
    baseline = json.loads((BASELINE_DIR / 'metrics.json').read_text(encoding='utf-8'))
    baseline_hashes = {p.name: file_hash(p) for p in BASELINE_DIR.iterdir() if p.is_file()}
    for name, digest in baseline['artifact_sha256'].items():
        if baseline_hashes[name] != digest:
            raise ValueError(f'Baseline artifact changed: {name}')
    for name, digest in baseline['code_sha256'].items():
        if file_hash(PROJECT_ROOT / name) != digest:
            raise ValueError(f'Baseline data/model code changed: {name}')
    for package, pinned in baseline['environment']['packages'].items():
        if version(package) != pinned:
            raise ValueError(f'Package drift: {package}')
    contract = PROJECT_ROOT / baseline['preprocessing_contract']['path']
    if file_hash(contract) != baseline['preprocessing_contract']['sha256']:
        raise ValueError('Frozen preprocessing contract changed')
    candidates = sample_candidates()  # Entire budget drawn before inspecting candidate scores.
    print('Loading unchanged development cohorts; holdout excluded...', flush=True)
    x_train, x_val, y_train, y_val = load_model_inputs()
    assert x_train.columns.tolist() == baseline['feature_names']
    assert x_train.dtypes.equals(x_val.dtypes)
    assert x_train.index.intersection(x_val.index).empty
    assert len(x_train) == baseline['splits']['train']['rows']
    assert len(x_val) == baseline['splits']['validation']['rows']
    saved_predictions = pd.read_csv(BASELINE_DIR / 'predictions.csv', float_precision='round_trip')
    np.testing.assert_array_equal(saved_predictions.source_row.to_numpy(), x_val.index.to_numpy())
    np.testing.assert_array_equal(saved_predictions.actual_default.to_numpy(), y_val.to_numpy())
    baseline_probabilities = lgb.Booster(model_file=str(BASELINE_DIR / 'model.txt')).predict(x_val)
    np.testing.assert_allclose(baseline_probabilities, saved_predictions.predicted_probability, rtol=1e-12, atol=1e-15)
    baseline_scores = score(y_val, baseline_probabilities)
    for key, result in baseline_scores.items():
        assert abs(result - baseline['validation_metrics'][key]) < 1e-14
    load_seconds = perf_counter() - started
    rows, best = [], None
    for number, overrides in enumerate(candidates, start=1):
        params = {**baseline['classifier_parameters'], **overrides}
        model = lgb.LGBMClassifier(**params)
        history = {}
        trial_started = perf_counter()
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_names=['validation'],
                  callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False),
                             lgb.record_evaluation(history)])
        probabilities = model.predict_proba(x_val, num_iteration=model.best_iteration_)[:, 1]
        metrics = score(y_val, probabilities)
        row = {'trial': number, 'status': 'complete', **overrides, **metrics,
               'best_iteration': int(model.best_iteration_),
               'evaluated_iterations': len(history['validation']['auc']),
               'reached_estimator_cap': len(history['validation']['auc']) == params['n_estimators'],
               'elapsed_seconds': perf_counter() - trial_started}
        rows.append(row)
        if best is None or metrics['roc_auc'] > best['metrics']['roc_auc']:
            best = {'trial': number, 'parameters': model.get_params(deep=False),
                    'booster_parameters': model.booster_.params, 'metrics': metrics,
                    'best_iteration': int(model.best_iteration_),
                    'history': history, 'probabilities': probabilities.copy(),
                    'model_text': model.booster_.model_to_string(num_iteration=model.best_iteration_)}
        print(f"Trial {number:02d}/{TRIAL_COUNT}: AUC={metrics['roc_auc']:.6f}, best_iteration={model.best_iteration_}, seconds={row['elapsed_seconds']:.2f}", flush=True)
    for name, digest in baseline_hashes.items():
        assert file_hash(BASELINE_DIR / name) == digest
    assert file_hash(contract) == baseline['preprocessing_contract']['sha256']
    reloaded = lgb.Booster(model_str=best['model_text'])
    reloaded_probabilities = reloaded.predict(x_val)
    np.testing.assert_allclose(reloaded_probabilities, best['probabilities'], rtol=1e-12, atol=1e-15)
    assert reloaded.current_iteration() == best['best_iteration']
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / 'tuning_trials.csv', index=False, float_format='%.17g')
    disk_difference = save_candidate_model(reloaded, output_dir / 'candidate_model.txt', x_val, best['probabilities'])
    pd.DataFrame({'source_row': x_val.index, 'split': 'validation', 'actual_default': y_val.to_numpy(),
                  'predicted_probability': best['probabilities']}).to_csv(
                      output_dir / 'validation_predictions.csv', index=False, float_format='%.17g')
    report = {
        'stage': 4, 'status': 'candidate_frozen_pending_holdout',
        'method': '24 pre-drawn seeded random-search trials; no adaptive extension',
        'selection': 'maximum validation ROC-AUC; ties retain earlier trial; secondary metrics reported only',
        'search_seed': SEARCH_SEED, 'model_seed': baseline['seed'],
        'search_space': SEARCH_SPACE, 'trials_requested': TRIAL_COUNT, 'trials_completed': len(rows),
        'best_trial': best['trial'], 'best_iteration': best['best_iteration'],
        'parameters': best['parameters'], 'booster_parameters': best['booster_parameters'],
        'fixed_iteration_parameters_for_stage5': {**best['parameters'], 'n_estimators': best['best_iteration']},
        'early_stopping': baseline['early_stopping'], 'best_trial_evaluation_history': best['history'],
        'best_validation_metrics': best['metrics'], 'baseline_validation_metrics': baseline_scores,
        'validation_differences_tuned_minus_baseline': {k: best['metrics'][k]-baseline_scores[k] for k in baseline_scores},
        'improved_validation_auc': best['metrics']['roc_auc'] > baseline_scores['roc_auc'],
        'feature_count': x_train.shape[1], 'splits': baseline['splits'],
        'holdout_evaluated': False, 'preprocessing_contract': baseline['preprocessing_contract'],
        'raw_source_sha256': baseline['source']['sha256'], 'baseline_artifact_sha256': baseline_hashes,
        'unchanged_baseline_code_sha256': baseline['code_sha256'],
        'tuning_code_sha256': file_hash(Path(__file__)), 'environment': baseline['environment'],
        'runtime_seconds': {'loading_and_baseline_verification': load_seconds,
                            'trials_total': sum(row['elapsed_seconds'] for row in rows),
                            'total_through_artifact_checks': perf_counter()-started},
        'verification': {'baseline_artifacts_unchanged': True, 'development_rows_match_baseline': True,
                         'model_reload_max_abs_difference': float(np.max(np.abs(reloaded_probabilities-best['probabilities']))),
                         'disk_model_reload_max_abs_difference': disk_difference},
        'limitations': baseline['limitations'] + ['Selecting from 24 trials adds validation selection optimism; improvement on holdout is unknown.'],
        'artifact_sha256': {name: file_hash(output_dir/name) for name in
                            ['tuning_trials.csv', 'candidate_model.txt', 'validation_predictions.csv']},
    }
    (output_dir / 'best_params.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['best_trial', 'best_iteration', 'best_validation_metrics',
                      'validation_differences_tuned_minus_baseline', 'runtime_seconds']}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    run_tuning()
