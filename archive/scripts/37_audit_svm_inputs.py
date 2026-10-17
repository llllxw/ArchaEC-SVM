"""Read-only audit of SVM splits, training-only feature selection and scaling."""
from pathlib import Path
import importlib, json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'results/svm_migration_20261008';OLD=ROOT/'results/five_fold_cv_20261007'

def main():
    # Audit original grouping source again, writing only to the new run.
    source=importlib.import_module('31_verify_cv_inputs')
    source.RUN=RUN
    data=pd.read_csv(ROOT/'data/processed/archaea_swissprot_ec_master.csv',dtype={'taxonomy_id':str})
    x=np.load(OLD/'features.npy');checks=[];rankchecks=[];scalerchecks=[];mappings={}
    for path in sorted((RUN/'splits').glob('*.csv')):
        split=pd.read_csv(path,dtype={'group':str});assert split.accession.tolist()==data.accession.tolist()
        assert split.accession.is_unique and set(split.outer_fold)==set(range(5))
        if path.stem.startswith('homology_'):
            level=path.stem.split('_')[1]
            clusters=pd.read_csv(ROOT/f'results/3.3_generalization_ladder/homology_id{level}/split_manifest.csv').set_index('accession')
            base=data.accession.map(clusters.homology_cluster_id);assert base.notna().all()
            mappings[path.stem]=pd.factorize(base)[0]
        else:base=data.taxonomy_id if path.stem=='species' else data.sequence_sha256
        for f in range(5):
            role=split[f'role_fold_{f}'];assert role.eq('test').equals(split.outer_fold.eq(f))
            for grouping in [base,data.sequence_sha256,split.group]:
                assert pd.DataFrame({'group':grouping,'role':role}).groupby('group').role.nunique().max()==1
            checks.append({'design':path.stem,'fold':f,'group_overlap':0,'sequence_overlap':0})
            task=RUN/'tasks'/f'svm__{path.stem}__{f}'
            if not (task/'complete.json').exists():continue
            tr=role.eq('train').to_numpy();cols=np.load(task/'columns.npy');pack=joblib.load(task/'model.joblib')
            assert np.allclose(pack['scaler'].mean_,x[tr][:,cols].mean(axis=0,dtype=np.float64),rtol=0,atol=1e-10)
            assert np.allclose(pack['scaler'].var_,x[tr][:,cols].var(axis=0,dtype=np.float64),rtol=1e-10,atol=1e-8)
            assert int(pack['scaler'].n_samples_seen_)==int(tr.sum())
            assert len(pack['models'])==7
            for model in pack['models']:
                assert model.C==2 and model.gamma=='scale' and model.class_weight=='balanced' and model.probability is False
                assert model.n_features_in_==len(cols) and model.support_.max()<tr.sum()
            scalerchecks.append({'design':path.stem,'fold':f,'train_only_scaler_verified':True})
            rankdir=OLD/'tasks'/f'rank__random__{f}' if path.stem=='random' else task
            rank=np.load(rankdir/'ranking.npy');assert sorted(rank.tolist())==list(range(432))
            edges=json.loads((rankdir/'mrmr_discretization.json').read_text())
            train=x[tr]
            for j in range(432):assert np.allclose(np.unique(np.quantile(train[:,j],np.arange(1,10)/10)),edges[j],rtol=0,atol=1e-12)
            if path.stem!='random':
                choices=[json.loads((task/f'prefix_{k}/validation.json').read_text()) for k in [16,32,64,128,256,432]]
                yv=data.loc[role.eq('validation'),[f'label_ec{i}' for i in range(1,8)]].to_numpy()
                for record in choices:
                    pv=np.load(task/f'prefix_{record["n_features"]}/validation_predictions.npy')
                    assert np.isclose(roc_auc_score(yv,pv,average='macro'),record['validation_macro_auc'],rtol=0,atol=1e-12)
                best=sorted(choices,key=lambda r:(-r['validation_macro_auc'],r['n_features']))[0]
                assert np.array_equal(cols,rank[:best['n_features']])
            rankchecks.append({'design':path.stem,'fold':f,'inner_train_quantile_edges_verified':True})
    identical=[];names=list(mappings)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if np.array_equal(mappings[a],mappings[b]):identical.append([a,b])
    shapchecks=[];explained_all=[];random=pd.read_csv(RUN/'splits/random.csv');index=pd.Index(data.accession)
    ablation_checks=[]
    feature_sets={'aac':np.arange(20),'dpc':np.arange(20,420),'phys':np.arange(420,432),
        'aac_dpc':np.arange(420),'aac_phys':np.r_[0:20,420:432],'dpc_phys':np.arange(20,432)}
    for f in range(5):
        tr=random[f'role_fold_{f}'].eq('train').to_numpy()
        for tag,expected in feature_sets.items():
            task=RUN/'tasks'/f'ablation__random__{f}__{tag}'
            if not (task/'complete.json').exists():continue
            cols=np.load(task/'columns.npy');assert np.array_equal(cols,expected)
            pack=joblib.load(task/'model.joblib');scaler=pack['scaler']
            assert int(scaler.n_samples_seen_)==int(tr.sum())
            assert np.allclose(scaler.mean_,x[tr][:,cols].mean(axis=0,dtype=np.float64),rtol=0,atol=1e-10)
            assert np.allclose(scaler.var_,x[tr][:,cols].var(axis=0,dtype=np.float64),rtol=1e-10,atol=1e-8)
            assert len(pack['models'])==7
            for model in pack['models']:
                assert model.C==2 and model.gamma=='scale' and model.class_weight=='balanced' and model.probability is False
                assert model.n_features_in_==len(cols) and model.support_.max()<tr.sum()
            ablation_checks.append({'fold':f,'feature_set':tag,'train_only_scaler_and_columns_verified':True})
    for f in range(5):
        task=RUN/'tasks'/f'shap__random__{f}'
        if not (task/'complete.json').exists():continue
        sampling=json.loads((task/'sampling_manifest.json').read_text());bg=sampling['background_accessions'];ex=sampling['explained_accessions']
        assert len(bg)==len(set(bg))==14 and len(ex)==len(set(ex))==42
        role=random[f'role_fold_{f}'];assert set(bg)<=set(data.loc[role.eq('train'),'accession'])
        assert set(ex)<=set(data.loc[role.eq('test'),'accession']);explained_all+=ex
        for ids,n in [(bg,2),(ex,6)]:
            sub=data.set_index('accession').loc[ids]
            assert sub.n_top_classes.eq(1).all() and sub.fragment.fillna('').ne('fragment').all()
            assert (sub[[f'label_ec{i}' for i in range(1,8)]].sum()==n).all()
        modeldir=RUN/'tasks'/f'svm__random__{f}';pack=joblib.load(modeldir/'model.joblib');cols=np.load(modeldir/'columns.npy')
        def margin(ids):
            arr=pack['scaler'].transform(x[index.get_indexer(ids)][:,cols]).astype(np.float32)
            return np.column_stack([model.decision_function(arr) for model in pack['models']])
        values=np.load(task/'shap_values.npy');assert values.shape==(42,432,7)
        error=float(np.max(np.abs(values.sum(axis=1)+margin(bg).mean(axis=0)-margin(ex))))
        assert error<1e-6
        shapchecks.append({'fold':f,'background_train_only':True,'explanations_test_only':True,
            'class_balanced_counts_verified':True,'additivity_recomputed_using_original_libsvm':error})
    assert len(explained_all)==len(set(explained_all))
    result={'passed':True,'split_checks':checks,'completed_ranking_checks':rankchecks,'kernelshap_checks':shapchecks,
        'ablation_scaler_checks':ablation_checks,
        'scaler_checks':scalerchecks,'identical_cluster_partitions':identical,
        'audit_scope':'independent recomputation of split overlap, training quantiles/scaler, selection rule; no retraining'}
    (RUN/'independent_input_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PASS:',len(checks),'fold splits;',len(rankchecks),'rankings;',len(scalerchecks),'scalers',flush=True)

if __name__=='__main__':main()
