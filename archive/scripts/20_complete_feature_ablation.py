"""Recover AP for historical ablations and add AAC+Phys/DPC+Phys.
Run supervisor has a 30 minute budget. Original seeds/config preserved.
"""
import sys,time,json,subprocess,importlib.util,shutil,os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/3.4_multifunction_interpretability'
RUN=OUT/'feature_ablation_complete_20260924'
LABELS=[f'label_ec{i}' for i in range(1,8)]

def worker():
    spec=importlib.util.spec_from_file_location('ablation',ROOT/'scripts/12_run_multifunction_interpretability.py')
    source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    master=pd.read_csv(ROOT/'data/processed/archaea_swissprot_ec_master.csv')
    cache=ROOT/'data/processed/aac_dpc_features.npy'
    assert cache.with_suffix('.accessions.txt').read_text().splitlines()==master.accession.tolist()
    x=np.hstack([np.load(cache),source.physicochemical_features(master.sequence)])
    manifest=pd.read_csv(ROOT/'results/3.2_random_split_performance/split_manifest.csv').set_index('accession').split_random
    split=master.accession.map(manifest)
    assert split.notna().all()
    train=split.eq('train').to_numpy();val=split.eq('validation').to_numpy();test=split.eq('test').to_numpy()
    y=master[LABELS].to_numpy(dtype=int)
    groups=[('AAC only','aac',np.arange(20),1042),
            ('DPC only','dpc',np.arange(20,420),1062),
            ('AAC + DPC','aac_dpc',np.arange(420),1082),
            ('Physicochemical only','phys',np.arange(420,432),1102),
            ('AAC + Physicochemical','aac_phys',np.r_[0:20,420:432],1122),
            ('DPC + Physicochemical','dpc_phys',np.arange(20,432),1142),
            ('Full model','full',np.arange(432),42)]
    previous=pd.read_csv(RUN/'previous_feature_ablation_auc.csv').set_index('feature_set')
    rows=[];classes=[]
    for name,tag,cols,seed in groups:
        p=np.zeros((test.sum(),7));folder=RUN/'models'/tag;folder.mkdir(parents=True,exist_ok=True)
        for c in range(7):
            if tag=='full':
                model=joblib.load(ROOT/f'results/3.2_random_split_performance/optimized_lightgbm/lightgbm_ec{c+1}.joblib')
            else:
                model=source.build_model(seed+c)
                model.fit(x[train][:,cols],y[train,c],eval_set=[(x[val][:,cols],y[val,c])],
                          eval_metric='auc',callbacks=[lgb.early_stopping(120,verbose=False),lgb.log_evaluation(0)])
                joblib.dump(model,folder/f'ec{c+1}.joblib')
            p[:,c]=model.predict_proba(x[test][:,cols],num_iteration=model.best_iteration_)[:,1]
            classes.append({'feature_set':name,'ec_class':c+1,'auc':roc_auc_score(y[test,c],p[:,c]),
                'ap':average_precision_score(y[test,c],p[:,c]),'seed':model.get_params()['random_state'],
                'best_iteration':int(model.best_iteration_)})
            print(f'{tag} EC{c+1} complete',flush=True)
        prediction=master.loc[test,['accession']+LABELS].copy()
        for c in range(7):prediction[f'score_ec{c+1}']=p[:,c]
        prediction.to_csv(RUN/'predictions'/f'{tag}_test.csv',index=False)
        row={'feature_set':name,'tag':tag,'n_features':len(cols),'seed_base':seed,
             'macro_roc_auc':roc_auc_score(y[test],p,average='macro'),
             'micro_roc_auc':roc_auc_score(y[test],p,average='micro'),
             'macro_ap':average_precision_score(y[test],p,average='macro'),
             'micro_ap':average_precision_score(y[test],p,average='micro')}
        row['previous_auc_difference']=float(row['macro_roc_auc']-previous.loc[name,'macro_roc_auc']) if name in previous.index else np.nan
        rows.append(row)
        pd.DataFrame(rows).to_csv(RUN/'feature_ablation_auc_ap.csv',index=False)
        pd.DataFrame(classes).to_csv(RUN/'per_class_auc_ap.csv',index=False)
        print('RESULT',json.dumps(row),flush=True)
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'feature_ablation_auc_ap.csv',index=False)
    table.to_csv(OUT/'feature_ablation_auc.csv',index=False)

def main():
    if '--worker' in sys.argv: worker();return
    if RUN.exists():raise FileExistsError('Existing run: inspect before retrying')
    RUN.mkdir();(RUN/'predictions').mkdir()
    shutil.copy2(OUT/'feature_ablation_auc.csv',RUN/'previous_feature_ablation_auc.csv')
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker']
    state={'status':'running','command':command,'budget_seconds':1800,
           'seed_policy':'Original four seed bases retained; two new bases 1122/1142; full model reused',
           'metric_definitions':'macro AUC = mean seven OvR AUCs; macro AP = mean seven average_precision scores'}
    start=time.monotonic()
    with (RUN/'training.log').open('w',encoding='utf-8') as log:
        env={**os.environ,'PYTHONIOENCODING':'utf-8'}
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        state['pid']=process.pid;(RUN/'run_status.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        try:
            code=process.wait(timeout=1800)
            state['status']='completed' if code==0 else 'failed';state['exit_code']=code
        except subprocess.TimeoutExpired:
            print('HARD TIMEOUT: stopping training process',process.pid,flush=True)
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
            state['status']='timeout'
    state['elapsed_seconds']=time.monotonic()-start
    (RUN/'run_status.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    print(json.dumps(state),flush=True)
    if state['status']!='completed':sys.exit(1)

if __name__=='__main__':main()
