"""Independent read-only audit of persisted CV partitions and completed IFS."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'results/five_fold_cv_20261007'

def main():
    master=pd.read_csv(ROOT/'data/processed/archaea_swissprot_ec_master.csv',dtype={'taxonomy_id':str})
    labels=[f'label_ec{i}' for i in range(1,8)];checks=[];cluster_hashes={};group_mappings={}
    for path in sorted((RUN/'splits').glob('*.csv')):
        split=pd.read_csv(path,dtype={'group':str});assert split.accession.tolist()==master.accession.tolist()
        assert set(split.outer_fold)==set(range(5)) and split.accession.is_unique
        if path.stem.startswith('homology_'):
            level=path.stem.split('_')[1];source=pd.read_csv(ROOT/f'results/3.3_generalization_ladder/homology_id{level}/split_manifest.csv').set_index('accession')
            base=master.accession.map(source.homology_cluster_id)
            cluster_hashes[path.stem]=hashlib.sha256((ROOT/f'results/3.3_generalization_ladder/homology_id{level}/split_manifest.csv').read_bytes()).hexdigest()
            group_mappings[path.stem]=pd.factorize(base)[0]
        else:base=master.taxonomy_id if path.stem=='species' else master.sequence_sha256
        for f in range(5):
            role=split[f'role_fold_{f}'];assert role.eq('test').equals(split.outer_fold.eq(f))
            for grouping in [base,master.sequence_sha256]:
                audit=pd.DataFrame({'group':grouping,'role':role})
                assert audit.groupby('group').role.nunique().max()==1
            for part in ['train','validation','test']:
                frame=master[role.eq(part)];assert (frame[labels].nunique()==2).all()
            checks.append({'design':path.stem,'fold':f,'base_group_overlap':0,'sequence_overlap':0,'test_n':int(role.eq('test').sum())})
    x=np.load(RUN/'features.npy');random=pd.read_csv(RUN/'splits/random.csv');rankchecks=[]
    for f in range(5):
        folder=RUN/'tasks'/f'rank__random__{f}'
        if not (folder/'complete.json').exists():continue
        ranking=np.load(folder/'ranking.npy');assert sorted(ranking.tolist())==list(range(432))
        edges=json.loads((folder/'mrmr_discretization.json').read_text())
        train=x[random[f'role_fold_{f}'].eq('train')]
        for j in range(432):assert np.allclose(np.unique(np.quantile(train[:,j],np.arange(1,10)/10)),edges[j],rtol=0,atol=1e-12)
        rankchecks.append({'fold':f,'quantile_edges_recomputed_from_inner_train_only':True})
    identical=[]
    names=list(group_mappings)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if np.array_equal(group_mappings[a],group_mappings[b]):identical.append([a,b])
    result={'passed':True,'split_checks':checks,'completed_ranking_checks':rankchecks,'cluster_source_hashes':cluster_hashes,
            'identical_cluster_partitions':identical,'audit_scope':'partitions and completed ranking quantile edges, no model rerun'}
    (RUN/'independent_input_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PASS',len(checks),'outer splits;',len(rankchecks),'completed training-only mRMR quantile audits')

if __name__=='__main__':main()
