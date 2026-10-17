"""Complete the 90..10% MMseqs2/LightGBM sweep without changing old 50/40 results.

Fixed seed/configuration; 1800-second overall budget; saved command logs and
split-validity checks. A failed process is recorded and never silently retried.
"""
import sys, time, json, subprocess, hashlib
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/3.3_generalization_ladder'
RUN=OUT/'expansion_20260924'
CLUSTERS=ROOT/'data/processed/homology_expansion_20260924'
MASTER=ROOT/'data/processed/archaea_swissprot_ec_master.csv'
LABELS=[f'label_ec{i}' for i in range(1,8)]
START=time.monotonic()
STATE={'status':'running','budget_seconds':1800,'algorithm':'LightGBM',
       'configuration':'scripts/07_run_lightgbm_auc_model.py; unchanged historical generalization configuration',
       'thresholds':list(range(90,0,-10)),'retained_thresholds':[50,40],'runs':[]}

def checkpoint():
    STATE['elapsed_seconds']=round(time.monotonic()-START,1)
    (RUN/'progress.json').write_text(json.dumps(STATE,indent=2),encoding='utf-8')

def execute(script,args,tag):
    remaining=1800-(time.monotonic()-START)
    if remaining<=0: raise TimeoutError('Overall 30 minute budget exhausted')
    command=[sys.executable,'-u',str(ROOT/'scripts'/script),*map(str,args)]
    print('START',tag,flush=True)
    record={'stage':tag,'command':command,'status':'running'}
    STATE['runs'].append(record); checkpoint()
    with (RUN/(tag+'.log')).open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        record['pid']=proc.pid; checkpoint()
        try: code=proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            print('HARD TIMEOUT: stopping experiment process tree',proc.pid,flush=True)
            subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
            record['status']='timeout';checkpoint();raise TimeoutError(tag)
    record['exit_code']=code; record['status']='completed' if code==0 else 'failed';checkpoint()
    if code: raise RuntimeError(f'{tag} failed with exit {code}; see log; no automatic retry')
    print('DONE',tag,flush=True)

def main():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'progress.json').exists(): raise FileExistsError('Run record exists; inspect before any rerun')
    checkpoint()
    data=pd.read_csv(MASTER)
    STATE['master_sha256']=hashlib.sha256(MASTER.read_bytes()).hexdigest()
    STATE['split_checks']=[]
    for threshold in [90,80,70,60,30,20,10]:
        execute('09_run_mmseqs_homology_clustering.py',[
            '--fasta',ROOT/'data/processed/archaea_swissprot_ec_master.fasta',
            '--mmseqs',ROOT/'tools/mmseqs2/mmseqs/bin/mmseqs',
            '--output-root',CLUSTERS,'--thresholds',threshold/100,
            '--threads',4,'--coverage',.8,'--sensitivity',7.5],f'id{threshold}_cluster')
        execute('10_create_homology_splits.py',['--master',MASTER,'--cluster-root',CLUSTERS,
            '--output-root',OUT,'--thresholds',threshold,'--seed',42],f'id{threshold}_split')
        folder=OUT/f'homology_id{threshold}'
        split=pd.read_csv(folder/'split_manifest.csv')
        merged=data.merge(split[['accession','homology_cluster_id','split_homology']],on='accession',validate='one_to_one')
        assert len(merged)==len(data)
        assert merged.groupby('homology_cluster_id').split_homology.nunique().max()==1
        invalid=[]
        for part in ['train','validation','test']:
            sub=merged.loc[merged.split_homology.eq(part)]
            for col in LABELS+['is_multifunctional_top_level']:
                if sub[col].nunique()!=2: invalid.append(f'{part}:{col}')
        check={'threshold':threshold,'invalid_binary_targets':invalid,'eligible':not invalid}
        STATE['split_checks'].append(check);checkpoint()
        if invalid:
            print('NOT EVALUABLE',threshold,invalid,flush=True)
            continue
        execute('07_run_lightgbm_auc_model.py',['--master',MASTER,
            '--split-manifest',folder/'split_manifest.csv','--split-column','split_homology',
            '--feature-cache',ROOT/'data/processed/aac_dpc_features.npy',
            '--output-dir',folder/'optimized_lightgbm','--seed',42],f'id{threshold}_train')
        pred=pd.read_csv(folder/'optimized_lightgbm/optimized_test_probabilities.csv')
        test=merged.loc[merged.split_homology.eq('test')]
        assert set(pred.accession)==set(test.accession)
        joined=pred.merge(data[['accession']+LABELS],on='accession',validate='one_to_one')
        auc=roc_auc_score(joined[LABELS],joined[[f'prob_ec{i}' for i in range(1,8)]],average='macro')
        recorded=pd.read_csv(folder/'optimized_lightgbm/optimized_auc_metrics.csv').iloc[0].macro_roc_auc
        assert abs(auc-recorded)<1e-12
        check['verified_macro_auc']=float(auc);checkpoint()
        print('VERIFIED',threshold,round(auc,6),flush=True)
    previous=pd.read_csv(OUT/'homology_auc_summary.csv')
    previous.to_csv(RUN/'previous_homology_auc_summary.csv',index=False)
    rows=previous.loc[~previous.evaluation.str.startswith('Homology-')].to_dict('records')
    for threshold in range(90,0,-10):
        metric=OUT/f'homology_id{threshold}/optimized_lightgbm/optimized_auc_metrics.csv'
        if metric.exists(): rows.append({'evaluation':f'Homology-{threshold}',**pd.read_csv(metric).iloc[0].to_dict()})
        else: rows.append({'evaluation':f'Homology-{threshold}','status':'not_evaluable'})
    pd.DataFrame(rows).to_csv(OUT/'homology_auc_summary.csv',index=False)
    STATE['status']='completed';checkpoint()
    print('SWEEP COMPLETED',flush=True)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        STATE['status']='failed';STATE['error']=str(exc)
        if RUN.exists(): checkpoint()
        raise
