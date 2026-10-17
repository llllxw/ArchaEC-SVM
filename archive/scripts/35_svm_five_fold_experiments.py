"""Approved SVM migration: fixed folds, inner mRMR-IFS, resumable six-hour run."""
from __future__ import annotations
import argparse, hashlib, importlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'results/five_fold_cv_20261007'
RUN=ROOT/'results/svm_migration_20261008'
SEED=20261007
LABELS=[f'label_ec{i}' for i in range(1,8)]
DESIGNS=['random','species']+[f'homology_{i}' for i in range(10,100,10)]
SETS={'aac':np.arange(20),'dpc':np.arange(20,420),'phys':np.arange(420,432),
      'aac_dpc':np.arange(420),'aac_phys':np.r_[0:20,420:432],'dpc_phys':np.arange(20,432)}
comp=importlib.import_module('14_mrmr_model_comparison')
old=importlib.import_module('26_five_fold_experiments')
dump=old.dump;log=old.log

def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def checkpoint_model(path,model):
    tmp=path.with_suffix('.joblib.tmp');joblib.dump(model,tmp);tmp.replace(path)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def folder(task):return RUN/'tasks'/task
def complete(task):
    p=folder(task)
    dump(p/'complete.json',{'task':task,'status':'complete','time':time.time(),
         'outputs':{str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file() and f.name not in ['complete.json','failure.json'] and not f.name.endswith('.tmp')}})

def prepare():
    for name,h in read(OLD/'config.json')['fingerprints'].items():assert sha(ROOT/name)==h,name
    assert read(OLD/'independent_input_audit.json')['passed']
    RUN.mkdir(exist_ok=True)
    config={'model':{'name':'RBF-SVM','C':2,'gamma':'scale','class_weight':'balanced','probability':False},
      'seed':SEED,'outer_folds':5,'inner_validation':'existing grouped holdout within each outer train',
      'ifs_grid':comp.GRID,'selection':'inner-training mRMR; maximum inner-validation macro AUC, smaller-prefix tie-break',
      'ablation':'all features of each predefined group, no group-specific IFS',
      'score':'sigmoid decision function, NOT calibrated probability',
      'kernelshap':{'background_per_class':2,'explained_test_per_class':6,'nsamples':1024,'l1_reg':'num_features(20)',
                    'output':'raw SVM decision function, identity link','population':'complete single-label',
                    'sampling':'fixed seed, independent of predictions; 14 background and 42 explained rows per fold'},
      'features_sha256':sha(OLD/'features.npy'),'master_sha256':sha(ROOT/'data/processed/archaea_swissprot_ec_master.csv'),
      'split_hashes':{p.name:sha(p) for p in (OLD/'splits').glob('*.csv')},
      'template_hashes':{str(n):sha(ROOT/f'手稿/结果图/图{n}.png') for n in [3,4,6]},
      'source_script_hash':sha(ROOT/'scripts/14_mrmr_model_comparison.py'),'external_validation':False}
    if (RUN/'config.json').exists():assert read(RUN/'config.json')==config,'Configuration drift'
    else:dump(RUN/'config.json',config)
    (RUN/'splits').mkdir(exist_ok=True)
    for p in (OLD/'splits').glob('*.csv'):
        q=RUN/'splits'/p.name
        if q.exists():assert sha(q)==sha(p)
        else:shutil.copy2(p,q)
    shutil.copy2(OLD/'fold_sample_counts.csv',RUN/'fold_sample_counts.csv')
    shutil.copy2(OLD/'split_audit.json',RUN/'split_audit.json')

def pack_fit(path,x,y,tr,va,cols,seed):
    path.mkdir(parents=True,exist_ok=True)
    sp=path/'scaler.joblib'
    if sp.exists():scaler=joblib.load(sp)
    else:scaler=StandardScaler().fit(x[tr][:,cols]);checkpoint_model(sp,scaler)
    xt=scaler.transform(x[tr][:,cols]).astype(np.float32)
    xv=scaler.transform(x[va][:,cols]).astype(np.float32)
    models=[];pv=[]
    for c in range(y.shape[1]):
        mp=path/f'ec{c+1}.joblib'
        if mp.exists():model=joblib.load(mp)
        else:
            model=SVC(C=2,gamma='scale',class_weight='balanced',cache_size=512,random_state=seed)
            model.fit(xt,y[tr,c]);checkpoint_model(mp,model)
        models.append(model);pv.append(expit(model.decision_function(xv)))
        log(f'{path.parent.name}/{path.name} EC{c+1} saved')
    pack={'scaler':scaler,'models':models,'name':'RBF-SVM'}
    return pack,np.asarray(pv).T

def save_main(task,data,x,y,tr,va,te,pack,cols,pv,selection,auxiliary):
    path=folder(task);pt=comp.predict(pack,x[te][:,cols])
    cuts=comp.thresholds(y[va],pv)
    metric=comp.metrics(y[te],pt,cuts);metric['micro_roc_auc']=roc_auc_score(y[te],pt,average='micro')
    checkpoint_model(path/'model.joblib',pack)
    np.save(path/'validation_predictions.npy',pv)
    if auxiliary:
        target=data.is_multifunctional_top_level.astype(int).to_numpy();ap=path/'auxiliary.joblib'
        if ap.exists():aux=joblib.load(ap)
        else:
            aux=SVC(C=2,gamma='scale',class_weight='balanced',cache_size=512,random_state=SEED+int(task.split('__')[2])*100+100)
            aux.fit(pack['scaler'].transform(x[tr][:,cols]).astype(np.float32),target[tr]);checkpoint_model(ap,aux)
        avt=expit(aux.decision_function(pack['scaler'].transform(x[te][:,cols]).astype(np.float32)))
        avv=expit(aux.decision_function(pack['scaler'].transform(x[va][:,cols]).astype(np.float32)))
        np.save(path/'validation_auxiliary.npy',avv)
        metric.update(auxiliary_roc_auc=roc_auc_score(target[te],avt),auxiliary_ap=average_precision_score(target[te],avt))
        pt=np.c_[pt,avt]
    old.save_predictions(data,y,te,pt,path,cuts,metric,selection=selection,model='RBF-SVM',score_type='uncalibrated sigmoid decision score')
    np.save(path/'columns.npy',np.asarray(cols))

class FastRBF:
    """Numerically audited equivalent of seven libsvm RBF decision functions."""
    def __init__(self,pack):
        self.scaler=pack['scaler'];models=pack['models']
        self.gamma=models[0]._gamma
        assert all(np.isclose(a._gamma,self.gamma,rtol=0,atol=1e-15) for a in models)
        ids=np.unique(np.concatenate([a.support_ for a in models]));self.sv=np.zeros((len(ids),models[0].n_features_in_))
        self.coeff=np.zeros((len(ids),len(models)));self.intercept=np.array([a.intercept_[0] for a in models])
        for c,a in enumerate(models):
            ix=np.searchsorted(ids,a.support_);self.sv[ix]=a.support_vectors_;self.coeff[ix,c]=a.dual_coef_[0]
        self.norm=np.einsum('ij,ij->i',self.sv,self.sv)
    def __call__(self,x):
        scaled=self.scaler.transform(np.asarray(x)).astype(np.float32).astype(np.float64);out=[]
        for start in range(0,len(x),256):
            a=scaled[start:start+256];d=np.einsum('ij,ij->i',a,a)[:,None]+self.norm[None,:]-2*a@self.sv.T
            kernel=np.exp(-self.gamma*np.maximum(d,0));out.append(kernel@self.coeff+self.intercept)
        return np.concatenate(out)

def run_shap(task,data,x,y,tr,te,f):
    sys.path.insert(0,str(ROOT/'.svm_dependencies'))
    import shap
    path=folder(task);src=folder(f'svm__random__{f}')
    pack=joblib.load(src/'model.joblib');cols=np.load(src/'columns.npy');fn=FastRBF(pack)
    samples=x[np.r_[tr[:12],te[:12]]][:,cols]
    ref=np.column_stack([a.decision_function(pack['scaler'].transform(samples).astype(np.float32)) for a in pack['models']])
    with threadpool_limits(limits=4):err=float(np.max(np.abs(ref-fn(samples))))
    assert err<1e-8,err
    rng=np.random.default_rng(SEED+9000+f)
    eligible=data.n_top_classes.eq(1)&data.fragment.fillna('').ne('fragment')
    background=[];explained=[]
    for c in range(7):
        bt=tr[eligible.iloc[tr].to_numpy()&(y[tr,c]==1)];et=te[eligible.iloc[te].to_numpy()&(y[te,c]==1)]
        assert len(bt)>=2 and len(et)>=6
        background.extend(rng.choice(bt,2,replace=False).tolist());explained.extend(rng.choice(et,6,replace=False).tolist())
    manifest={'fold':f,'background_accessions':data.accession.iloc[background].tolist(),'explained_accessions':data.accession.iloc[explained].tolist(),
        'n_background':len(background),'n_explained':len(explained),'seed':SEED+9000+f,'nsamples':1024,'l1_reg':'num_features(20)',
        'accelerated_predictor_max_absolute_error':err,'shap_version':shap.__version__,'output':'decision function, not probability'}
    dump(path/'sampling_manifest.json',manifest)
    vals=[];residuals=[]
    with threadpool_limits(limits=4):
        explainer=shap.KernelExplainer(fn,x[background][:,cols],link='identity')
        for i,row in enumerate(explained):
            cp=path/f'sample_{i:03d}.npy'
            if cp.exists():v=np.load(cp)
            else:
                np.random.seed(SEED+9000+f*1000+i)
                v=explainer.shap_values(x[[row]][:,cols],nsamples=1024,l1_reg='num_features(20)',silent=True)[0]
                assert v.shape==(len(cols),7)
                np.save(cp,v)
            residual=float(np.max(np.abs(v.sum(axis=0)+explainer.expected_value-fn(x[[row]][:,cols])[0])))
            assert residual<1e-6,residual
            vals.append(v);residuals.append(residual);log(f'{task}: explanation {i+1}/42, additive residual {residual:.2e}')
    values=np.asarray(vals);full=np.zeros((len(explained),432,7));full[:,cols,:]=values
    np.save(path/'shap_values.npy',full)
    sums=[]
    for c in range(7):
        pos=y[explained,c].astype(bool);v=full[:,:,c]
        sums.append(np.c_[np.abs(v).mean(axis=0),v[pos].mean(axis=0)-v[~pos].mean(axis=0)])
    np.save(path/'shap_summary.npy',np.array(sums));dump(path/'additivity_audit.json',{'max_absolute_residual':max(residuals),'passed':True})

def worker(task):
    p=folder(task);p.mkdir(parents=True,exist_ok=True)
    if (p/'failure.json').exists():raise RuntimeError('Prior failure requires explicit inspected retry: '+task)
    try:
        data=old.load();x=np.load(OLD/'features.npy');y=data[LABELS].to_numpy(int)
        kind,design,fold,*suffix=task.split('__');f=int(fold);comp.SEED=SEED+f*100
        role=pd.read_csv(RUN/'splits'/f'{design}.csv')[f'role_fold_{f}'].to_numpy()
        tr,va,te=[np.flatnonzero(role==s) for s in ['train','validation','test']]
        if kind=='shap':run_shap(task,data,x,y,tr,te,f)
        elif kind=='ablation':
            cols=SETS[suffix[0]]
            with threadpool_limits(limits=2):pack,pv=pack_fit(p/'fit',x,y,tr,va,cols,comp.SEED)
            save_main(task,data,x,y,tr,va,te,pack,cols,pv,{'columns':cols.tolist(),'n_features':len(cols),'feature_set':suffix[0]},False)
        elif design=='random':
            source=OLD/'tasks'/f'selected__random__{f}__RBF-SVM';rec=read(source/'metrics.json');selection=rec['selection']
            original=OLD/rec['model_path'];record=read(original/'complete.json')
            for name,h in record['outputs'].items():assert sha(original/name)==h
            pack=joblib.load(original/'model.joblib');cols=selection['columns'];pv=np.load(original/'validation_predictions.npy')
            save_main(task,data,x,y,tr,va,te,pack,cols,pv,selection,True)
            previous=pd.read_csv(source/'predictions.csv',float_precision='round_trip')
            now=pd.read_csv(p/'predictions.csv',float_precision='round_trip')
            assert np.array_equal(previous[[f'score_ec{i}' for i in range(1,8)]],now[[f'score_ec{i}' for i in range(1,8)]])
            dump(p/'reuse.json',{'source':str(source.relative_to(ROOT)),'predictions_identical':True,'auxiliary_newly_trained':True})
        else:
            rp=p/'ranking.npy'
            if rp.exists():ranking=np.load(rp)
            else:
                comp.OUT=p;ranking=comp.mrmr(x[tr],y[tr]);np.save(rp,ranking)
            choices=[]
            for k in comp.GRID:
                sub=p/f'prefix_{k}';vp=sub/'validation.json';cols=ranking[:k]
                if vp.exists():record=read(vp)
                else:
                    with threadpool_limits(limits=2):pack,pv=pack_fit(sub,x,y,tr,va,cols,comp.SEED)
                    checkpoint_model(sub/'model.joblib',pack);np.save(sub/'validation_predictions.npy',pv)
                    record={'n_features':k,'validation_macro_auc':float(roc_auc_score(y[va],pv,average='macro')),'columns':cols.tolist()};dump(vp,record)
                choices.append(record);log(f'{task}: IFS {k} validation={record["validation_macro_auc"]:.6f}')
            best=sorted(choices,key=lambda a:(-a['validation_macro_auc'],a['n_features']))[0];sub=p/f'prefix_{best["n_features"]}'
            save_main(task,data,x,y,tr,va,te,joblib.load(sub/'model.joblib'),best['columns'],np.load(sub/'validation_predictions.npy'),best,True)
        complete(task);log('DONE '+task)
    except Exception as exc:
        dump(p/'failure.json',{'task':task,'error':str(exc),'traceback':traceback.format_exc(),'automatic_retry':False});raise

def tasks():
    return [f'svm__{d}__{f}' for d in DESIGNS for f in range(5)]+[f'ablation__random__{f}__{s}' for f in range(5) for s in SETS]+[f'shap__random__{f}' for f in range(5)]

def supervise(hours,resume):
    import psutil
    sp=RUN/'run_status.json';previous=read(sp) if sp.exists() else {}
    if previous and not resume:raise FileExistsError('Run exists; inspect before --resume')
    if previous.get('status')=='running' and psutil.pid_exists(previous.get('pid',-1)):raise RuntimeError('Supervisor already active')
    start=time.monotonic();used=previous.get('elapsed_seconds',0);deadline=hours*3600
    state={'status':'running','pid':os.getpid(),'started_at':time.strftime('%Y-%m-%d %H:%M:%S'),'budget_seconds':deadline,'previous_elapsed_seconds':used,
           'active':{},'completed':[],'errors':[],'elapsed_seconds':used}
    dump(sp,state);prepare();jobs=tasks();dump(RUN/'task_schedule.json',jobs);pending=[]
    for task in jobs:
        cp=folder(task)/'complete.json'
        if cp.exists():
            for n,h in read(cp)['outputs'].items():assert sha(folder(task)/n)==h
            state['completed'].append(task)
        else:pending.append(task)
    live={};(RUN/'logs').mkdir(exist_ok=True)
    while pending or live:
        state['elapsed_seconds']=used+time.monotonic()-start
        if state['elapsed_seconds']>=deadline:
            log('HARD TIMEOUT: six hours reached; stopping workers, checkpoints retained')
            for proc,_ in live.values():subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
            state['status']='timeout';dump(sp,state);return
        for task,(proc,handle) in list(live.items()):
            if proc.poll() is not None:
                handle.close();del live[task]
                if proc.returncode:state['errors'].append({'task':task,'exit_code':proc.returncode});pending=[]
                else:assert (folder(task)/'complete.json').exists();state['completed'].append(task)
        while pending and len(live)<2 and not state['errors']:
            task=pending.pop(0);handle=(RUN/'logs'/f'{task}.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',task],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
            live[task]=(proc,handle);log('START '+task)
        state['active']={t:{'pid':p.pid} for t,(p,h) in live.items()};state['completed_count']=len(state['completed']);state['total_tasks']=len(jobs)
        dump(sp,state)
        if live:time.sleep(10)
    state['status']='failed' if state['errors'] else 'training_complete';state['elapsed_seconds']=used+time.monotonic()-start;dump(sp,state)
    log(state['status'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--hours',type=float,default=6);parser.add_argument('--resume',action='store_true');parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.prepare_only:prepare()
    else:supervise(args.hours,args.resume)
