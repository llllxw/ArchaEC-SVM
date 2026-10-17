"""Audited grouped outer five-fold evaluation, inner holdout, resumable tasks.

Run: python -u scripts/26_five_fold_experiments.py --hours 6
Resume: same command --resume (budget is cumulative, not reset).
Original scientific scripts and result files remain untouched during training.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, platform, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/five_fold_cv_20261007'
LABELS=[f'label_ec{i}' for i in range(1,8)]
SEED=20261007
DESIGNS=['random','species']+[f'homology_{i}' for i in range(10,100,10)]
SETS={'full':np.arange(432),'aac':np.arange(20),'dpc':np.arange(20,420),
      'phys':np.arange(420,432),'aac_dpc':np.arange(420),
      'aac_phys':np.r_[0:20,420:432],'dpc_phys':np.arange(20,432)}

def module(file,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(path)

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def log(s): print(time.strftime('%Y-%m-%d %H:%M:%S'),s,flush=True)
def load():
    data=pd.read_csv(ROOT/'data/processed/archaea_swissprot_ec_master.csv',dtype={'taxonomy_id':str})
    assert data.accession.is_unique
    return data

def union_groups(data,base):
    """Transitive union of requested groups and exact-sequence duplicates."""
    codes=pd.factorize(np.asarray(base,dtype=str))[0];parent=np.arange(codes.max()+1)
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]];x=parent[x]
        return x
    for ids in data.groupby('sequence_sha256',sort=False).indices.values():
        root=find(codes[ids[0]])
        for i in ids[1:]: parent[find(codes[i])]=root
    return np.array([str(find(i)) for i in codes])

def balanced_group_folds(groups,y,k,seed):
    """Greedy multilabel group balancing, fixed seed, no performance selection.

    Minimize incremental squared normalized fold loads for positive/negative
    label counts plus sample counts. Large rare-label groups placed first.
    """
    unique,inv=np.unique(groups,return_inverse=True)
    counts=np.zeros((len(unique),y.shape[1]*2+1))
    sample=np.c_[y,1-y,np.ones(len(y))]
    np.add.at(counts,inv,sample)
    totals=counts.sum(axis=0);norm=counts/np.maximum(totals,1)
    rng=np.random.default_rng(seed);noise=rng.uniform(0,1e-10,len(unique))
    order=np.argsort(-(norm.max(axis=1)+noise))
    loads=np.zeros((k,norm.shape[1]));assignment=np.zeros(len(unique),int)
    for g in order:
        costs=((loads+norm[g])**2-loads**2).sum(axis=1)
        candidates=np.flatnonzero(np.isclose(costs,costs.min(),rtol=0,atol=1e-15))
        f=int(rng.choice(candidates));loads[f]+=norm[g];assignment[g]=f
    return assignment[inv]

def prepare():
    data=load();RUN.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/'data/processed/archaea_swissprot_ec_master.csv',ROOT/'data/processed/aac_dpc_features.npy',
             ROOT/'scripts/07_run_lightgbm_auc_model.py',ROOT/'scripts/14_mrmr_model_comparison.py']
    fingerprints={str(p.relative_to(ROOT)):digest(p) for p in sources}
    config={'seed':SEED,'outer_folds':5,'inner_validation':'one grouped fold of 5 within outer training',
            'sd_ddof':1,'feature_selection':'fit mRMR only on inner train; select IFS by inner validation AUC',
            'metrics':'macro seven-class ROC-AUC/AP; accuracy exact multilabel match; threshold from inner validation',
            'group_policy':'transitive union with identical sequence; grouped label-load balancing',
            'fingerprints':fingerprints,'python':platform.python_version(),'designs':DESIGNS,
            'comparison_config':'14_mrmr_model_comparison.py unchanged',
            'downstream_config':'07_run_lightgbm_auc_model.py unchanged',
            'external_validation':False,'previously_inspected_dataset':True}
    if (RUN/'config.json').exists():
        assert json.loads((RUN/'config.json').read_text())==config,'Input/config changed: do not reuse checkpoints'
    else:dump(RUN/'config.json',config)
    y=np.c_[data[LABELS].to_numpy(int),data.is_multifunctional_top_level.astype(int).to_numpy()]
    audits=[];foldstats=[]
    for design in DESIGNS:
        if design=='random': base=data.sequence_sha256
        elif design=='species':base=data.taxonomy_id
        else:
            ident=design.split('_')[1]
            source=ROOT/f'results/3.3_generalization_ladder/homology_id{ident}/split_manifest.csv'
            frame=pd.read_csv(source).set_index('accession');base=data.accession.map(frame.homology_cluster_id)
            assert base.notna().all()
        groups=union_groups(data,base)
        folds=balanced_group_folds(groups,y,5,SEED)
        assignments=pd.DataFrame({'accession':data.accession,'group':groups,'outer_fold':folds.astype(np.int64)})
        for f in range(5):
            te=np.flatnonzero(folds==f);remaining=np.flatnonzero(folds!=f)
            inside=balanced_group_folds(groups[remaining],y[remaining],5,SEED+100+f)
            va=remaining[inside==0];tr=remaining[inside!=0]
            roles=np.full(len(data),'train',dtype='<U10');roles[te]='test';roles[va]='validation'
            assignments[f'role_fold_{f}']=roles
            gsets=[set(groups[ix]) for ix in [tr,va,te]]
            assert not(gsets[0]&gsets[1] or gsets[0]&gsets[2] or gsets[1]&gsets[2])
            seqsets=[set(data.sequence_sha256.iloc[ix]) for ix in [tr,va,te]]
            assert not(seqsets[0]&seqsets[1] or seqsets[0]&seqsets[2] or seqsets[1]&seqsets[2])
            for role,ix in [('train',tr),('validation',va),('test',te)]:
                pos=y[ix].sum(axis=0);neg=len(ix)-pos
                invalid=[i+1 for i in range(8) if pos[i]==0 or neg[i]==0]
                foldstats.append({'design':design,'fold':f,'role':role,'n':len(ix),'groups':len(set(groups[ix])),
                                  **{f'positive_ec{i+1}':int(pos[i]) for i in range(7)},'positive_aux':int(pos[7]),'invalid_targets':str(invalid)})
                audits.append({'design':design,'fold':f,'role':role,'invalid_targets':invalid,'group_overlap':0,'sequence_overlap':0})
        path=RUN/'splits'/f'{design}.csv';path.parent.mkdir(exist_ok=True)
        if path.exists():pd.testing.assert_frame_equal(pd.read_csv(path,dtype={'group':str}),assignments)
        else:assignments.to_csv(path,index=False)
    pd.DataFrame(foldstats).to_csv(RUN/'fold_sample_counts.csv',index=False)
    dump(RUN/'split_audit.json',{'passed':all(not a['invalid_targets'] for a in audits),'checks':audits})
    # Never produce a fake seven-class macro metric by silently dropping a label.
    invalid=[a for a in audits if a['invalid_targets']]
    if invalid:raise ValueError(f'Non-estimable grouped folds: {invalid}; intervention required before training')
    src=module('07_run_lightgbm_auc_model.py','cv_source_lgb')
    if not (RUN/'features.npy').exists():
        np.save(RUN/'features.npy',src.load_features(data,ROOT/'data/processed/aac_dpc_features.npy'))
    log('AUDIT PASSED: 11 designs x 5 outer folds; zero group and duplicate-sequence leakage')

def taskdir(task):return RUN/'tasks'/task
def fit_lgb(data,x,y,split,folder,columns,seed,auxiliary=True):
    src=module('07_run_lightgbm_auc_model.py','cv_source_lgb')
    comp=module('14_mrmr_model_comparison.py','cv_source_compare')
    tr,va,te=[np.flatnonzero(split==s) for s in ['train','validation','test']]
    targets=np.c_[y,data.is_multifunctional_top_level.astype(int)] if auxiliary else y
    pv=[];pt=[];iterations=[]
    for c in range(targets.shape[1]):
        path=folder/f'ec{c+1}.joblib'
        if path.exists():model=joblib.load(path)
        else:
            model=src.build_model(seed+c if c<7 else seed+100)
            model.fit(x[tr][:,columns],targets[tr,c],eval_set=[(x[va][:,columns],targets[va,c])],eval_metric='auc',
                      callbacks=[lgb.early_stopping(120,verbose=False),lgb.log_evaluation(0)])
            temp=path.with_suffix('.joblib.tmp');joblib.dump(model,temp);temp.replace(path)
        pv.append(model.predict_proba(x[va][:,columns])[:,1]);pt.append(model.predict_proba(x[te][:,columns])[:,1]);iterations.append(int(model.best_iteration_))
        log(f'{folder.name} class {c+1}/{targets.shape[1]} saved')
    pv=np.array(pv).T;pt=np.array(pt).T
    cuts=comp.thresholds(y[va],pv[:,:7]);metrics=comp.metrics(y[te],pt[:,:7],cuts)
    metrics.update(micro_roc_auc=roc_auc_score(y[te],pt[:,:7],average='micro'))
    if auxiliary:metrics.update(auxiliary_roc_auc=roc_auc_score(targets[te,7],pt[:,7]),auxiliary_ap=average_precision_score(targets[te,7],pt[:,7]))
    save_predictions(data,y,te,pt,folder,cuts,metrics,iterations=iterations)

def save_predictions(data,y,te,pt,folder,cuts,metrics,**meta):
    pred=data.iloc[te][['accession']+LABELS].copy()
    for c in range(7):pred[f'score_ec{c+1}']=pt[:,c]
    if pt.shape[1]>7:pred['score_aux']=pt[:,7];pred['label_aux']=data.iloc[te].is_multifunctional_top_level.astype(int).to_numpy()
    pred.to_csv(folder/'predictions.csv',index=False)
    pc=[{'ec_class':c+1,'roc_auc':roc_auc_score(y[te,c],pt[:,c]),'ap':average_precision_score(y[te,c],pt[:,c])} for c in range(7)]
    pd.DataFrame(pc).to_csv(folder/'per_class.csv',index=False)
    dump(folder/'metrics.json',{'metrics':{k:float(v) for k,v in metrics.items()},'thresholds':cuts.tolist(),**meta})

def worker(task):
    folder=taskdir(task);folder.mkdir(parents=True,exist_ok=True)
    log('TASK '+task);data=load();x=np.load(RUN/'features.npy');y=data[LABELS].to_numpy(int)
    parts=task.split('__');kind=parts[0];design=parts[1];f=int(parts[2])
    split=pd.read_csv(RUN/'splits'/f'{design}.csv')[f'role_fold_{f}'].to_numpy()
    tr,va,te=[np.flatnonzero(split==s) for s in ['train','validation','test']]
    seed=SEED+f*100
    if kind in ['lgb','ablation']:
        columns=SETS[parts[3]] if kind=='ablation' else SETS['full']
        fit_lgb(data,x,y,split,folder,columns,seed,kind=='lgb')
    elif kind=='rank':
        comp=module('14_mrmr_model_comparison.py','cv_source_compare');comp.OUT=folder
        np.save(folder/'ranking.npy',comp.mrmr(x[tr],y[tr]))
    elif kind=='compare':
        comp=module('14_mrmr_model_comparison.py','cv_source_compare');comp.OUT=folder;comp.SEED=seed
        name=parts[3].replace('_',' ');k=int(parts[4]);cols=np.load(taskdir(f'rank__random__{f}')/'ranking.npy')[:k]
        values=comp.encode_sequences(data.sequence) if name=='1D-CNN' else x[:,cols]
        with threadpool_limits(limits=4):pack=comp.train_model(name,values[tr],y[tr],values[va],y[va])
        pv=comp.predict(pack,values[va]);valscore=float(roc_auc_score(y[va],pv,average='macro'))
        # Only validation predictions at each prefix; outer test remains unqueried until selection.
        np.save(folder/'validation_predictions.npy',pv)
        joblib.dump(pack,folder/'model.joblib')
        dump(folder/'validation.json',{'model':name,'n_features':k,'fold':f,'validation_macro_auc':valscore,'columns':cols.tolist()})
    elif kind=='selected':
        comp=module('14_mrmr_model_comparison.py','cv_source_compare');comp.OUT=folder;comp.SEED=seed
        name=parts[3].replace('_',' ')
        choices=[taskdir(f'compare__random__{f}__{parts[3]}__{k}') for k in ([0] if name=='1D-CNN' else comp.GRID)]
        records=[json.loads((p/'validation.json').read_text()) for p in choices]
        best=sorted(zip(records,choices),key=lambda a:(-a[0]['validation_macro_auc'],a[0]['n_features']))[0]
        rec,path=best;pack=joblib.load(path/'model.joblib');cols=rec['columns']
        values=comp.encode_sequences(data.sequence) if name=='1D-CNN' else x[:,cols]
        pv=np.load(path/'validation_predictions.npy');pt=comp.predict(pack,values[te])
        cuts=comp.thresholds(y[va],pv)
        save_predictions(data,y,te,pt,folder,cuts,comp.metrics(y[te],pt,cuts),selection=rec,model_path=str(path.relative_to(RUN)))
    elif kind=='shap':
        # Only outer-test, complete single-label proteins, matching prior interpretation population.
        mask=(data.iloc[te].n_top_classes.eq(1)&data.iloc[te].fragment.fillna('').ne('fragment')).to_numpy()
        ix=te[mask];values=[]
        for c in range(7):
            model=joblib.load(taskdir(f'lgb__random__{f}')/f'ec{c+1}.joblib')
            vals=model.booster_.predict(x[ix],pred_contrib=True,num_iteration=model.best_iteration_)[:,:-1]
            pos=y[ix,c].astype(bool)
            values.append(np.c_[np.abs(vals).mean(axis=0),vals[pos].mean(axis=0)-vals[~pos].mean(axis=0)])
        np.save(folder/'shap_summary.npy',np.array(values));dump(folder/'population.json',{'n':len(ix),'accessions':data.accession.iloc[ix].tolist()})
    else:raise ValueError(task)
    dump(folder/'complete.json',{'task':task,'status':'complete','time':time.time(),
                               'outputs':{p.name:digest(p) for p in folder.iterdir() if p.is_file() and p.name not in ['complete.json','task.lock']}})
    log('DONE '+task)

def locked_worker(task):
    """An exclusive task lease allows a second process lane, not duplicate fits."""
    import psutil
    folder=taskdir(task);folder.mkdir(parents=True,exist_ok=True);lease=folder/'task.lock'
    while True:
        if (folder/'complete.json').exists():return
        if (folder/'failure.json').exists():
            raise RuntimeError(f'Prior task failure requires inspection and explicit retry authorization: {task}')
        try:
            fd=os.open(str(lease),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            with os.fdopen(fd,'w') as handle:handle.write(str(os.getpid()))
            break
        except FileExistsError:
            content=lease.read_text().strip()
            if content and not psutil.pid_exists(int(content)):
                raise RuntimeError(f'Stale task lease {lease}; inspect failed task before explicitly resuming')
            time.sleep(5)
    try:worker(task)
    except Exception as exc:
        dump(folder/'failure.json',{'task':task,'error':str(exc),'traceback':traceback.format_exc(),'automatic_retry':False})
        raise
    finally:
        if lease.exists() and lease.read_text().strip()==str(os.getpid()):lease.unlink()

def tasklist():
    jobs=[f'lgb__{d}__{f}' for d in DESIGNS for f in range(5)]
    jobs += [f'ablation__random__{f}__{s}' for f in range(5) for s in SETS if s!='full']
    jobs += [f'shap__random__{f}' for f in range(5)]
    comp=module('14_mrmr_model_comparison.py','cv_source_compare')
    for f in range(5):
        jobs.append(f'rank__random__{f}')
        for name in comp.NAMES:
            tag=name.replace(' ','_')
            jobs.extend(f'compare__random__{f}__{tag}__{k}' for k in ([0] if name=='1D-CNN' else comp.GRID))
            jobs.append(f'selected__random__{f}__{tag}')
    return jobs

def supervisor(hours,resume):
    RUN.mkdir(parents=True,exist_ok=True);statepath=RUN/'run_status.json'
    if statepath.exists() and not resume:raise FileExistsError('Run exists; use --resume only after inspection')
    previous=json.loads(statepath.read_text()) if statepath.exists() else {}
    if previous.get('status')=='running':
        import psutil
        if psutil.pid_exists(previous.get('supervisor_pid',-1)):raise RuntimeError('Supervisor still running')
    used=float(previous.get('elapsed_seconds',0));start=time.monotonic()
    state={'status':'running','supervisor_pid':os.getpid(),'started_at':time.strftime('%Y-%m-%d %H:%M:%S'),
           'budget_seconds':hours*3600,'previous_elapsed_seconds':used,'completed':previous.get('completed',[]),'current_task':'prepare'}
    def checkpoint():state['elapsed_seconds']=used+time.monotonic()-start;dump(statepath,state)
    checkpoint()
    try:
        prepare();jobs=tasklist();state['total_tasks']=len(jobs);dump(RUN/'task_schedule.json',jobs)
        for task in jobs:
            complete=taskdir(task)/'complete.json'
            if complete.exists():
                rec=json.loads(complete.read_text())
                assert all((taskdir(task)/p).exists() and digest(taskdir(task)/p)==h for p,h in rec['outputs'].items()),'Checkpoint integrity failed'
                if task not in state['completed']:state['completed'].append(task)
                continue
            remaining=hours*3600-used-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError('Cumulative six-hour budget exhausted; checkpoints retained')
            state['current_task']=task;checkpoint();(RUN/'logs').mkdir(exist_ok=True)
            with (RUN/'logs'/f'{task}.log').open('w',encoding='utf-8') as logf:
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',task],stdout=logf,stderr=subprocess.STDOUT,
                                      env={**os.environ,'PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
                state['worker_pid']=proc.pid;checkpoint();last_size=-1;stall=0
                while proc.poll() is None:
                    time.sleep(10);checkpoint()
                    if state['elapsed_seconds']>=hours*3600:
                        log(f'HARD TIMEOUT: terminating worker {proc.pid}, checkpoints retained')
                        subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
                        raise TimeoutError('Six-hour limit reached')
                    size=(RUN/'logs'/f'{task}.log').stat().st_size
                    stall=stall+1 if size==last_size else 0;last_size=size
                    state['output_stall_seconds']=stall*10
                    # Advisory only; a long model fit is not terminated for quiet logs.
                if proc.returncode:raise RuntimeError(f'{task}: exit={proc.returncode}; inspect log, no automatic retry')
            assert complete.exists();state['completed'].append(task);checkpoint();log(f'PROGRESS {len(state["completed"])}/{len(jobs)} {task}')
        state['status']='training_complete';state['current_task']='ready_for_aggregation';checkpoint()
    except TimeoutError as e:state['status']='timeout';state['error']=str(e);checkpoint();raise
    except Exception as e:state['status']='failed';state['error']=str(e);checkpoint();raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--hours',type=float,default=6)
    args=parser.parse_args()
    if args.worker:locked_worker(args.worker)
    elif args.prepare_only:prepare()
    else:supervisor(args.hours,args.resume)
