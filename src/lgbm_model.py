"""Fit and freeze the Stage 3 baseline without evaluating the final holdout."""
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
from importlib.metadata import version

try:
    from .stratified_sampling import load_development_data, read_contract, VALIDATION_PATH, SEED
    from .feature_screening import transform_features
except ImportError:
    from stratified_sampling import load_development_data, read_contract, VALIDATION_PATH, SEED
    from feature_screening import transform_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline"


def load_model_inputs():
    train, validation, report = load_development_data()
    schema = report['preprocessing']
    return (
        transform_features(train, schema), transform_features(validation, schema),
        train['target'], validation['target'],
    )


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_lgbm_model(output_dir=OUTPUT_DIR):
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Baseline output already exists; do not overwrite the frozen baseline")
    started = perf_counter()
    contract_hash = file_hash(VALIDATION_PATH)
    report = read_contract()
    print("Loading frozen training and validation cohorts (holdout excluded)...", flush=True)
    X_train, X_validation, y_train, y_validation = load_model_inputs()
    if file_hash(VALIDATION_PATH) != contract_hash:
        raise ValueError("Data contract changed during loading")
    assert X_train.columns.tolist() == report['preprocessing']['features']
    assert X_train.dtypes.equals(X_validation.dtypes)
    for name, target in [('train', y_train), ('validation', y_validation)]:
        assert len(target) == report['splits'][name]['rows']
        assert int(target.sum()) == report['splits'][name]['positive_rows']
    load_seconds = perf_counter() - started

    print("Training the unchanged Stage 2 baseline configuration...", flush=True)
    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        class_weight=None,
        metric='auc',
        random_state=SEED,
        n_jobs=-1
    )
    history = {}
    fit_started = perf_counter()
    clf.fit(
        X_train, y_train,
        eval_set=[(X_validation, y_validation)],
        eval_names=['validation'],
        callbacks=[lgb.early_stopping(stopping_rounds=50, first_metric_only=True),
                   lgb.record_evaluation(history)]
    )
    fit_seconds = perf_counter() - fit_started
    best_iteration = int(clf.best_iteration_)
    probabilities = clf.predict_proba(X_validation, num_iteration=best_iteration)[:, 1]
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Invalid predicted probabilities")
    validation_metrics = {
        'roc_auc': float(roc_auc_score(y_validation, probabilities)),
        'average_precision': float(average_precision_score(y_validation, probabilities)),
        'log_loss': float(log_loss(y_validation, probabilities, labels=[0, 1])),
        'brier_score': float(brier_score_loss(y_validation, probabilities)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / 'model.txt'
    clf.booster_.save_model(str(model_path), num_iteration=best_iteration)
    reloaded = lgb.Booster(model_file=str(model_path))
    reloaded_probabilities = reloaded.predict(X_validation)
    np.testing.assert_allclose(reloaded_probabilities, probabilities, rtol=1e-12, atol=1e-15)
    assert reloaded.current_iteration() == best_iteration

    predictions = pd.DataFrame({
        'source_row': X_validation.index,
        'split': 'validation',
        'actual_default': y_validation.to_numpy(),
        'predicted_probability': probabilities,
    })
    predictions.to_csv(output_dir / 'predictions.csv', index=False, float_format='%.17g')
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'gain': clf.booster_.feature_importance(importance_type='gain', iteration=best_iteration),
        'split_count': clf.booster_.feature_importance(importance_type='split', iteration=best_iteration),
    }).sort_values(['gain', 'feature'], ascending=[False, True]).reset_index(drop=True)
    importance.insert(0, 'rank', np.arange(1, len(importance) + 1))
    importance.to_csv(output_dir / 'feature_importance.csv', index=False, float_format='%.17g')
    code_files = ['src/lgbm_model.py', 'src/stratified_sampling.py',
                  'src/feature_screening.py', 'src/feature_engineering.py']
    packages = ['lightgbm', 'scikit-learn', 'pandas', 'numpy', 'scipy',
                'joblib', 'threadpoolctl', 'narwhals', 'cloudpickle']
    metrics = {
        'stage': 3,
        'status': 'frozen',
        'evaluation_split': 'validation',
        'validation_metrics': validation_metrics,
        'splits': report['splits'],
        'holdout': {'rows': report['splits']['holdout']['rows'], 'evaluated': False,
                    'default_rate': None, 'metrics': None},
        'feature_count': X_train.shape[1],
        'feature_names': X_train.columns.tolist(),
        'seed': SEED,
        'best_iteration': best_iteration,
        'evaluated_iterations': len(history['validation']['auc']),
        'early_stopping': {'rounds': 50, 'metric': 'auc', 'first_metric_only': True},
        'classifier_parameters': clf.get_params(deep=False),
        'booster_parameters': clf.booster_.params,
        'evaluation_history': history,
        'source': report['source'],
        'policy': report['policy'],
        'preprocessing_contract': {'path': str(VALIDATION_PATH.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                                   'sha256': contract_hash},
        'code_sha256': {path: file_hash(PROJECT_ROOT / path) for path in code_files},
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'packages': {package: version(package) for package in packages}},
        'runtime_seconds': {'load_and_preprocess': load_seconds, 'fit': fit_seconds,
                            'total_through_artifact_checks': perf_counter() - started},
        'verification': {'saved_model_prediction_max_abs_difference': float(np.max(np.abs(reloaded_probabilities - probabilities))),
                         'saved_model_iterations': reloaded.current_iteration()},
        'historical_comparison': {
            'readme_roc_auc': 0.6885,
            'validation_auc_minus_readme': validation_metrics['roc_auc'] - 0.6885,
            'directly_comparable': False,
            'reason': 'Different cohort, target treatment, prevalence, preprocessing and split; historical score lacks saved run evidence.',
        },
        'prediction_row_id': 'Zero-based raw CSV record position, not a loan/customer ID',
        'limitations': ['Resolved 36-month historical cohort only; not a prospective backtest.',
                        'Customer overlap cannot be verified because IDs are absent.',
                        'Bureau snapshot timing remains an assumption.',
                        'Validation was used for early stopping; final holdout performance is unknown.'],
        'artifact_sha256': {name: file_hash(output_dir / name) for name in
                            ['model.txt', 'predictions.csv', 'feature_importance.csv']},
    }
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'validation_metrics': validation_metrics, 'best_iteration': best_iteration,
                      'fit_seconds': fit_seconds, 'model_reload_verified': True,
                      'output_directory': str(output_dir)}, indent=2), flush=True)
    return clf


if __name__ == '__main__':
    train_lgbm_model()
