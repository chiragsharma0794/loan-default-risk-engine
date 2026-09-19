"""Stage 7: export frozen holdout results and narrow reporting dimensions only."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    from .stratified_sampling import source_sha256
except ImportError:
    from stratified_sampling import source_sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/powerbi'
BAND_LABELS = ['Below 10%', '10% to below 15%', '15% to below 20%', '20% and above']
RAW_COLUMNS = ['loan_status', 'issue_d', 'term', 'loan_amnt', 'int_rate',
               'annual_inc', 'grade', 'sub_grade', 'purpose', 'home_ownership', 'addr_state']


def assign_risk_bands(probabilities):
    """Fixed reporting boundaries; independent of outcomes and classification."""
    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Probabilities must be finite and in [0, 1]')
    return np.searchsorted([0.10, 0.15, 0.20], values, side='right') + 1


def summarize_bands(loans):
    summary = loans.groupby('risk_band_id').agg(
        loan_count=('actual_default', 'size'), default_count=('actual_default', 'sum'),
        default_rate=('actual_default', 'mean'),
        avg_predicted_probability=('predicted_probability', 'mean'),
    ).reindex(range(1, 5))
    for name in ['loan_count', 'default_count']:
        summary[name] = summary[name].fillna(0).astype(int)
    summary['risk_band'] = BAND_LABELS
    return summary.reset_index()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run():
    require(not OUTPUT.exists(), 'Power BI output already exists; review before replacing it.')
    final = ROOT / 'outputs/final_model'
    explain = ROOT / 'outputs/explainability'
    metrics = json.loads((final / 'metrics.json').read_text())
    metadata = json.loads((explain / 'explanation_metadata.json').read_text())
    require(metrics['selected_model'] == 'tuned', 'Unexpected final model selection')
    inputs = {final / name: sha for name, sha in metrics['artifact_sha256'].items()}
    inputs[explain / 'shap_importance.csv'] = metadata['artifact_sha256']['shap_importance.csv']
    for path in [final / 'metrics.json', explain / 'explanation_metadata.json']:
        inputs[path] = source_sha256(path)
    for path, expected in inputs.items():
        require(source_sha256(path) == expected, f'Artifact hash mismatch: {path.name}')
    require(metadata['model_sha256'] == inputs[final / 'model.txt'], 'SHAP model mismatch')
    raw = ROOT / 'data/loan.csv'
    require(source_sha256(raw) == metrics['source_sha256'] == metadata['source_sha256'],
            'Raw source hash mismatch; source_row join unsafe')

    scores = pd.read_csv(final / 'holdout_predictions.csv', float_precision='round_trip')
    require(scores.source_row.is_unique and scores.source_row.notna().all(), 'Invalid score keys')
    selected = set(scores.source_row)
    pieces = []
    for chunk in pd.read_csv(raw, usecols=RAW_COLUMNS, chunksize=100_000, low_memory=False):
        subset = chunk.loc[chunk.index.isin(selected)].copy()
        if not subset.empty:
            subset.index.name = 'source_row'
            pieces.append(subset.reset_index())
    dimensions = pd.concat(pieces, ignore_index=True)
    loans = scores.merge(dimensions, on='source_row', how='left', validate='one_to_one', indicator=True)
    require(loans['_merge'].eq('both').all(), 'Missing reporting dimensions')
    require(loans.actual_default.eq(loans.loan_status.map({'Fully Paid': 0, 'Charged Off': 1})).all(),
            'Raw outcomes differ from saved outcomes')
    require(loans.term.str.strip().eq('36 months').all(), 'Unexpected term')
    issue = pd.to_datetime(loans.issue_d, format='%b-%Y', errors='raise')
    require(issue.dt.year.eq(2014).all(), 'Unexpected holdout dates')
    loans['issue_month'] = issue.dt.strftime('%Y-%m-%d')
    loans['predicted_probability'] = loans['tuned_probability']
    loans['predicted_class'] = (loans.predicted_probability >= metrics['threshold']).astype(int)
    loans['risk_band_id'] = assign_risk_bands(loans.predicted_probability)
    loans['risk_band'] = loans.risk_band_id.map(dict(enumerate(BAND_LABELS, 1)))
    loans['term_months'] = 36
    loans = loans.rename(columns={'loan_amnt': 'loan_amount', 'annual_inc': 'annual_income',
                                  'int_rate': 'interest_rate_pct', 'addr_state': 'state'})
    categorical = ['grade', 'sub_grade', 'purpose', 'home_ownership', 'state']
    for name in categorical:
        loans[name] = loans[name].astype('string').str.strip().replace('', pd.NA).fillna('Unknown')
    loans = loans[['source_row', 'actual_default', 'predicted_probability', 'predicted_class',
                   'risk_band_id', 'risk_band', 'issue_month', 'term_months', 'loan_amount',
                   'annual_income', 'interest_rate_pct', *categorical]].sort_values('source_row')
    require(len(loans) == metrics['holdout']['rows'], 'Row total mismatch')
    require(int(loans.actual_default.sum()) == metrics['holdout']['positive_rows'], 'Default total mismatch')
    np.testing.assert_allclose(loans.predicted_probability.mean(),
                               metrics['holdout_metrics']['tuned']['mean_probability'], rtol=0, atol=1e-14)

    performance = []
    for model, values in metrics['holdout_metrics'].items():
        row = {'model': model, 'is_final': int(model == metrics['selected_model']),
               'split': 'holdout_2014', 'loan_count': len(loans),
               'default_count': int(loans.actual_default.sum())}
        row.update({key: value for key, value in values.items() if key != 'confusion_matrix'})
        row.update(values['confusion_matrix'])
        performance.append(row)
    performance = pd.DataFrame(performance)
    # Explicit numeric null: CSV blanks must remain undefined, never become zero.
    performance['precision'] = pd.to_numeric(performance['precision'], errors='raise')
    booster = lgb.Booster(model_file=str(final / 'model.txt'))
    importance = pd.DataFrame({'feature': booster.feature_name(),
                               'importance': booster.feature_importance(importance_type='gain')})
    importance = importance.sort_values(['importance', 'feature'], ascending=[False, True])
    importance['importance_rank'] = np.arange(1, len(importance) + 1)
    shap = pd.read_csv(explain / 'shap_importance.csv', float_precision='round_trip')
    require(set(shap.feature) == set(importance.feature), 'Feature list mismatch')
    require(shap.feature.is_unique, 'Duplicate SHAP features')
    bands = pd.DataFrame({'risk_band_id': range(1, 5), 'risk_band': BAND_LABELS,
                          'lower_probability': [0, .1, .15, .2],
                          'upper_probability': [.1, .15, .2, 1], 'upper_inclusive': [0, 0, 0, 1]})
    tables = {'scored_loans': loans, 'risk_bands': bands, 'risk_band_summary': summarize_bands(loans),
              'model_performance': performance, 'feature_importance': importance,
              'shap_importance': shap,
              'calibration': pd.read_csv(final / 'calibration.csv', float_precision='round_trip')}
    OUTPUT.mkdir(parents=True)
    exported = {}
    for name, table in tables.items():
        path = OUTPUT / f'{name}.csv'
        table.to_csv(path, index=False, encoding='utf-8', float_format='%.17g')
        reread = pd.read_csv(path, float_precision='round_trip')
        pd.testing.assert_frame_equal(reread, table.reset_index(drop=True), check_dtype=False,
                                      check_exact=False, rtol=0, atol=1e-14)
        exported[name] = {'rows': len(table), 'bytes': path.stat().st_size, 'sha256': source_sha256(path)}
    saved = pd.read_csv(OUTPUT / 'scored_loans.csv', float_precision='round_trip')
    summary = pd.read_csv(OUTPUT / 'risk_band_summary.csv', float_precision='round_trip')
    pd.testing.assert_frame_equal(summarize_bands(saved), summary, check_dtype=False,
                                  check_exact=False, rtol=0, atol=1e-14)
    require(summary.loan_count.sum() == len(saved), 'Export count mismatch')
    require(summary.default_count.sum() == saved.actual_default.sum(), 'Export default mismatch')
    for path, expected in inputs.items():
        require(source_sha256(path) == expected, f'Frozen artifact changed: {path.name}')
    manifest = {'stage': 7, 'status': 'complete', 'source_sha256': metrics['source_sha256'],
                'export_script_sha256': source_sha256(Path(__file__)),
                'inputs': {str(path.relative_to(ROOT)): sha for path, sha in inputs.items()},
                'population': metrics['holdout'], 'model': 'tuned', 'threshold': metrics['threshold'],
                'risk_band_method': 'Fixed 0.10, 0.15, 0.20 cut points; lower inclusive, upper exclusive except 1.0; reporting only',
                'shap_sample_rows': metadata['sampling']['global_rows'],
                'verification': 'Raw hash, one-to-one join, outcomes/dates/term, frozen totals/mean, all CSV round trips, exported band totals, unchanged input hashes passed',
                'missing_numeric_counts': {name: int(loans[name].isna().sum()) for name in ['loan_amount', 'annual_income', 'interest_rate_pct']},
                'unknown_category_counts': {name: int(loans[name].eq('Unknown').sum()) for name in categorical},
                'tables': exported}
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'tables': exported, 'risk_bands': summary.to_dict('records'),
                      'csv_bytes': sum(item['bytes'] for item in exported.values())}, indent=2))


if __name__ == '__main__':
    run()
