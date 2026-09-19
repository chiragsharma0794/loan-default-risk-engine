"""Stage 6: explain the saved final model; never fit or select a model."""
import os
from pathlib import Path
import json
from time import perf_counter
from importlib.metadata import version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / 'venv' / 'mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from scipy.special import expit

try:
    from .lgbm_model import file_hash
    from .stratified_sampling import RAW_DATA_PATH, VALIDATION_PATH, read_contract, source_sha256
    from .feature_screening import transform_features, EXCLUDED_COLUMNS
except ImportError:
    from lgbm_model import file_hash
    from stratified_sampling import RAW_DATA_PATH, VALIDATION_PATH, read_contract, source_sha256
    from feature_screening import transform_features, EXCLUDED_COLUMNS

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'explainability'
SAMPLE_SIZE = 10000
SEED = 42


def choose_local_examples(predictions):
    """Nearest observed score to whole-cohort P10/P90, within each actual class."""
    rows = []
    for label, outcome in [(0, 'nondefault'), (1, 'default')]:
        for quantile, level in [(0.1, 'lower'), (0.9, 'higher')]:
            target = float(predictions.tuned_probability.quantile(quantile))
            candidates = predictions.loc[predictions.actual_default.eq(label)].copy()
            candidates['distance'] = (candidates.tuned_probability - target).abs()
            chosen = candidates.sort_values(['distance', 'source_row']).iloc[0]
            positive = bool(chosen.tuned_probability >= 0.5)
            classification = ('TP' if label else 'FP') if positive else ('FN' if label else 'TN')
            rows.append({'case': f'{level}_score_{outcome}', 'source_row': int(chosen.source_row),
                         'actual_default': label, 'predicted_class_at_0_5': int(positive),
                         'classification': classification, 'predicted_probability': float(chosen.tuned_probability),
                         'target_score_quantile': quantile, 'target_probability': target})
    return rows


def load_selected_features(row_ids, contract):
    schema = contract['preprocessing']
    columns = list(dict.fromkeys(schema['raw_features'] + ['issue_d', 'loan_status', 'term']))
    parts, metadata = [], []
    for chunk in pd.read_csv(RAW_DATA_PATH, usecols=columns, dtype='string', chunksize=50000):
        selected = chunk.loc[chunk.index.isin(row_ids)]
        if selected.empty:
            continue
        dates = pd.to_datetime(selected.issue_d, format='%b-%Y')
        assert dates.dt.year.eq(2014).all() and selected.term.str.strip().eq('36 months').all()
        parts.append(transform_features(selected, schema))
        metadata.append(pd.DataFrame({'issue_month': dates.dt.strftime('%Y-%m'),
                                      'actual_default': selected.loan_status.map({'Fully Paid':0,'Charged Off':1})}))
    features, meta = pd.concat(parts), pd.concat(metadata)
    assert features.index.is_unique and set(features.index) == set(row_ids)
    assert features.columns.tolist() == schema['features']
    return features, meta


def save_plot(path):
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_shap_plots(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Explainability outputs already exist; do not overwrite silently')
    started = perf_counter()
    final_dir = PROJECT_ROOT / 'outputs' / 'final_model'
    metrics = json.loads((final_dir / 'metrics.json').read_text())
    assert metrics['status'] == 'complete_holdout_consumed' and metrics['selected_model'] == 'tuned'
    frozen_hashes = {str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in final_dir.iterdir() if p.is_file()}
    for name, digest in metrics['artifact_sha256'].items():
        assert file_hash(final_dir / name) == digest
    for package, pinned in metrics['environment']['packages'].items():
        assert version(package) == pinned, f'Modeling dependency changed: {package}'
    assert file_hash(VALIDATION_PATH) == metrics['preprocessing_contract']['sha256']
    assert source_sha256(RAW_DATA_PATH) == metrics['source_sha256']
    contract = read_contract()
    predictions = pd.read_csv(final_dir / 'holdout_predictions.csv', float_precision='round_trip')
    assert predictions.source_row.is_unique and len(predictions) == metrics['holdout']['rows']
    global_ids = predictions.sample(n=SAMPLE_SIZE, random_state=SEED).source_row.to_numpy()
    examples = choose_local_examples(predictions)
    selected_ids = sorted(set(global_ids).union(row['source_row'] for row in examples))
    print(f'Loading {len(selected_ids):,} selected rows for a 10,000-row global sample plus local cases...', flush=True)
    X, metadata = load_selected_features(selected_ids, contract)
    assert set(X.columns).isdisjoint(EXCLUDED_COLUMNS)
    assert not any('hardship' in c for c in X.columns)
    expected = predictions.set_index('source_row').loc[X.index]
    np.testing.assert_array_equal(metadata.actual_default, expected.actual_default)
    model = lgb.Booster(model_file=str(final_dir / 'model.txt'))
    assert model.current_iteration() == 57 and model.feature_name() == X.columns.tolist()
    probabilities = model.predict(X)
    np.testing.assert_allclose(probabilities, expected.tuned_probability, rtol=1e-12, atol=1e-15)
    print('Computing path-dependent TreeSHAP in raw default log-odds...', flush=True)
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent', model_output='raw')
    values = explainer.shap_values(X, check_additivity=True)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 2:
        values = values[:, :, 1]
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[-1])
    assert values.shape == X.shape and np.isfinite(values).all()
    margins = model.predict(X, raw_score=True)
    reconstructed_margins = base_value + values.sum(axis=1)
    np.testing.assert_allclose(reconstructed_margins, margins, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(expit(reconstructed_margins), probabilities, rtol=1e-10, atol=1e-12)
    native_contributions = model.predict(X, pred_contrib=True)
    np.testing.assert_allclose(native_contributions[:, :-1], values, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(native_contributions[:, -1], base_value, rtol=1e-10, atol=1e-10)
    positions = X.index.get_indexer(global_ids)
    global_values, global_X = values[positions], X.loc[global_ids]
    importance = pd.DataFrame({'feature': X.columns, 'mean_abs_shap': np.abs(global_values).mean(axis=0)})
    importance = importance.sort_values(['mean_abs_shap','feature'], ascending=[False,True]).reset_index(drop=True)
    importance.insert(0,'importance_rank',np.arange(1,len(importance)+1))
    importance['importance_share'] = importance.mean_abs_shap / importance.mean_abs_shap.sum()
    output_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_dir/'shap_importance.csv', index=False, float_format='%.17g')
    np.savez_compressed(output_dir/'shap_values.npz', source_rows=X.index.to_numpy(),
                        feature_names=np.asarray(X.columns,dtype=str), values=values,
                        base_value=base_value, global_sample_rows=global_ids)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10,8))
    ax.barh(top.feature,top.mean_abs_shap,color='#237c9b')
    ax.set(xlabel='Mean absolute SHAP value (default log-odds)',title='Final model: global feature importance\n10,000 randomly sampled 2014 holdout loans')
    fig.tight_layout()
    save_plot(output_dir/'shap_importance.png')
    np.random.seed(SEED)
    shap.summary_plot(global_values, global_X, max_display=20, show=False, plot_size=(11,8))
    plt.title('Final model: SHAP summary\nPositive values increase predicted default log-odds',pad=18)
    plt.xlabel('SHAP value (default log-odds); categorical values shown in grey')
    save_plot(output_dir/'shap_summary.png')
    numeric = [c for c in importance.feature if pd.api.types.is_numeric_dtype(global_X[c])
               and global_X[c].nunique()>1 and importance.set_index('feature').loc[c,'mean_abs_shap']>0]
    effects = []
    for feature in numeric:
        column = global_X[feature].to_numpy(dtype=float)
        impacts = global_values[:,X.columns.get_loc(feature)]
        valid = np.isfinite(column)
        low,high = np.quantile(column[valid],[0.2,0.8])
        lower,upper = valid & (column<=low),valid & (column>=high)
        effects.append({'feature':feature,'low_20pct_cutoff':float(low),'high_20pct_cutoff':float(high),
                        'low_group_rows':int(lower.sum()),'high_group_rows':int(upper.sum()),
                        'low_group_mean_shap':float(impacts[lower].mean()),
                        'high_group_mean_shap':float(impacts[upper].mean()),
                        'high_minus_low_mean_shap':float(impacts[upper].mean()-impacts[lower].mean()),
                        'spearman_value_shap':float(pd.Series(column[valid]).corr(pd.Series(impacts[valid]),method='spearman'))})
    pd.DataFrame(effects).to_csv(output_dir/'numeric_effects.csv', index=False, float_format='%.17g')
    for feature in numeric[:3]:
        column = global_X[feature].to_numpy(dtype=float)
        valid = np.isfinite(column)
        impacts = global_values[:,X.columns.get_loc(feature)]
        fig,ax=plt.subplots(figsize=(9,5))
        ax.scatter(column[valid],impacts[valid],s=8,alpha=0.22,color='#237c9b',rasterized=True)
        ax.axhline(0,color='#555555',linewidth=0.8)
        positive = column[valid & (column>0)]
        scale_note=''
        if len(positive) and positive.max()>20*np.median(positive):
            ax.set_xscale('symlog',linthresh=max(float(np.median(positive))/10,1e-6))
            if column[valid].min() >= 0:
                ax.set_xlim(0, float(column[valid].max()) * 1.02)
            scale_note=' (symlog axis)'
        ax.set(xlabel=feature+scale_note,ylabel='SHAP value (default log-odds)',title=f'{feature}: observed model effect\nAssociations, not causal effects')
        fig.tight_layout();save_plot(output_dir/f'dependence_{feature}.png')
    categorical = [c for c in importance.feature if isinstance(global_X[c].dtype,pd.CategoricalDtype)
                   and importance.set_index('feature').loc[c,'mean_abs_shap']>0][:3]
    category_rows=[]
    for feature in categorical:
        table=pd.DataFrame({'category':global_X[feature].astype('string').fillna('<missing/unseen>'),
                            'shap_value':global_values[:,X.columns.get_loc(feature)]})
        grouped=table.groupby('category',sort=True).shap_value.agg(['size','mean'])
        for category,row in grouped.iterrows():
            category_rows.append({'feature':feature,'category':str(category),'rows':int(row['size']),'mean_shap':float(row['mean'])})
    categorical_effects=pd.DataFrame(category_rows)
    categorical_effects.to_csv(output_dir/'categorical_effects.csv',index=False,float_format='%.17g')
    if categorical:
        feature=categorical[0]
        common=categorical_effects[categorical_effects.feature.eq(feature)].nlargest(12,'rows').sort_values('mean_shap')
        fig,ax=plt.subplots(figsize=(10,6))
        ax.barh([f"{row.category} (n={row.rows})" for row in common.itertuples()],common.mean_shap,
                color=['#c55765' if value>0 else '#237c9b' for value in common.mean_shap])
        ax.axvline(0,color='#555555',linewidth=0.8)
        ax.set(xlabel='Mean SHAP value (default log-odds)',title=f'{feature}: 12 most frequent sampled categories\nBars ordered by mean SHAP, not by category label')
        fig.tight_layout();save_plot(output_dir/f'categorical_{feature}.png')
    local_values=[]
    for case in examples:
        index=X.index.get_loc(case['source_row'])
        explanation=shap.Explanation(values=values[index],base_values=base_value,
                                     data=X.iloc[index].to_numpy(),feature_names=X.columns.tolist())
        shap.plots.waterfall(explanation,max_display=10,show=False)
        plt.gcf().set_size_inches(12,7)
        plt.title(f"{case['case'].replace('_',' ')} | {case['classification']} at 0.5\nActual={case['actual_default']}; predicted default={case['predicted_probability']:.2%}; contributions in log-odds",pad=22)
        save_plot(output_dir/f"local_{case['case']}.png")
        order=np.argsort(-np.abs(values[index]),kind='stable')
        for rank,j in enumerate(order,start=1):
            value=X.iloc[index,j]
            local_values.append({'case':case['case'],'source_row':case['source_row'],'feature':X.columns[j],
                                 'feature_value':str(value) if pd.notna(value) else '<missing/unseen>',
                                 'shap_value':float(values[index,j]),'absolute_rank':rank})
        case['base_log_odds']=base_value
        case['reconstructed_probability']=float(expit(base_value+values[index].sum()))
    pd.DataFrame(examples).to_csv(output_dir/'local_examples.csv',index=False,float_format='%.17g')
    pd.DataFrame(local_values).to_csv(output_dir/'local_contributions.csv',index=False,float_format='%.17g')
    for name,digest in frozen_hashes.items(): assert file_hash(PROJECT_ROOT/name)==digest
    assert file_hash(VALIDATION_PATH)==metrics['preprocessing_contract']['sha256']
    sample_metadata=metadata.loc[global_ids]
    report={'stage':6,'model_path':'outputs/final_model/model.txt','model_sha256':file_hash(final_dir/'model.txt'),
            'model_iterations':model.current_iteration(),'feature_count':len(X.columns),
            'method':'TreeSHAP; tree_path_dependent; model_output=raw; no background refit',
            'units':'Default log-odds; positive SHAP increases modeled default risk, not a causal effect',
            'base_log_odds':base_value,'sampling':{'method':'simple random sample without replacement from final holdout predictions',
            'seed':SEED,'global_rows':SAMPLE_SIZE,'explained_rows_including_local_examples':len(X),
            'population_rows':len(predictions),'sample_positive_rows':int(sample_metadata.actual_default.sum()),
            'sample_default_rate':float(sample_metadata.actual_default.mean()),'population_default_rate':metrics['holdout']['default_rate'],
            'issue_month_counts':{str(k):int(v) for k,v in sample_metadata.issue_month.value_counts().sort_index().items()}},
            'local_selection':'Nearest score to full-cohort P10/P90 within each actual class; source-row tie-break; not added to global importance unless randomly sampled',
            'local_examples':examples,'unavailable_cases_at_0_5':['true_positive','false_positive'],
            'top_features':importance.head(10).to_dict('records'),'numeric_dependence_features':numeric[:3],
            'categorical_effect_features':categorical,'verification':{
                'shap_additivity_max_abs_error':float(np.max(np.abs(reconstructed_margins-margins))),
                'reconstructed_probability_max_abs_error':float(np.max(np.abs(expit(reconstructed_margins)-probabilities))),
                'saved_prediction_max_abs_error':float(np.max(np.abs(probabilities-expected.tuned_probability.to_numpy()))),
                'native_contribution_max_abs_error':float(np.max(np.abs(native_contributions[:,:-1]-values))),
                'final_artifacts_unchanged':True,'known_excluded_fields_absent':True},
            'source_sha256':metrics['source_sha256'],'preprocessing_contract':metrics['preprocessing_contract'],
            'frozen_final_artifact_sha256':frozen_hashes,'code_sha256':file_hash(Path(__file__)),
            'packages':{name:version(name) for name in ['shap','matplotlib','lightgbm','numpy','pandas','numba','llvmlite']},
            'runtime_seconds':perf_counter()-started,'limitations':metrics['limitations']+[
                'Global explanations describe a 10,000-row 2014 cohort sample, not the entire portfolio.',
                'Correlated geography, credit-vintage and pricing variables can share attribution; SHAP does not establish causation or absence of leakage.',
                'Path-dependent reference uses training tree path counts; no probability-point interpretation of raw SHAP values.'],
            'artifact_sha256':{p.name:file_hash(p) for p in output_dir.iterdir() if p.is_file()}}
    (output_dir/'explanation_metadata.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'top_features':report['top_features'],'sampling':report['sampling'],
                      'local_examples':examples,'verification':report['verification']},indent=2),flush=True)
    return report


if __name__=='__main__':
    generate_shap_plots()
