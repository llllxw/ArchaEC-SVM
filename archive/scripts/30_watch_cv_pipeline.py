"""Monitor approved training and stage complete five-fold figures automatically.

Does not retry failed training or publish unreviewed figures. Shares the original
six-hour wall-clock training deadline. Each plot stage runs at most once.
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'results/five_fold_cv_20261007'
STATE=RUN/'pipeline_monitor.json'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def atomic(obj):
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(STATE)
def main():
    if STATE.exists():raise FileExistsError('Monitor record exists. Inspect before restarting; no automatic retry')
    state={'status':'monitoring','pid':os.getpid(),'staged':[],'errors':[],'current_report':None,'resource_baselines':{}}
    proc=None;stage=None
    while True:
        primary=read(RUN/'run_status.json');side=read(RUN/'comparison_lane_status.json')
        completed={p.parent.name for p in (RUN/'tasks').glob('*/complete.json')}
        state.update(updated_at=time.strftime('%Y-%m-%d %H:%M:%S'),completed_tasks=len(completed),
                     total_tasks=primary.get('total_tasks',315),elapsed_seconds=primary['elapsed_seconds'],
                     primary_status=primary['status'],comparison_status=side['status'],
                     primary_task=primary.get('current_task'),comparison_task=side.get('current_task'))
        resource=[]
        for label,pid in [('primary',primary.get('worker_pid')),('comparison',side.get('worker_pid'))]:
            if pid and psutil.pid_exists(pid):
                try:
                    p=psutil.Process(pid);rss=p.memory_info().rss
                    base=state['resource_baselines'].setdefault(str(pid),rss)
                    resource.append({'lane':label,'pid':pid,'rss_bytes':rss,'cpu_seconds':sum(p.cpu_times()[:2]),'rss_vs_initial':rss/max(base,1)})
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        state['resources']=resource
        if proc is not None and proc.poll() is not None:
            if proc.returncode:state['errors'].append({'stage':stage,'exit_code':proc.returncode,'automatic_retry':False})
            else:state['staged'].append(stage)
            proc=None;state['current_report']=None;atomic(state)
        if primary['status'] in ['failed','timeout'] or side['status'] in ['failed','timeout']:
            state['status']='needs_attention';atomic(state);return
        remaining=primary['budget_seconds']-primary['elapsed_seconds']
        if remaining<=0:
            if proc is not None:
                print('HARD TIMEOUT: stopping report process',proc.pid,flush=True)
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
            state['status']='timeout';atomic(state);return
        needed={
          'generalization':{f'lgb__{d}__{f}' for d in ['random','species']+[f'homology_{i}' for i in range(10,100,10)] for f in range(5)},
          'ablation':{f'ablation__random__{f}__{s}' for f in range(5) for s in ['aac','dpc','phys','aac_dpc','aac_phys','dpc_phys']}|{f'lgb__random__{f}' for f in range(5)},
          'interpretability':{f'shap__random__{f}' for f in range(5)}|{f'lgb__homology_40__{f}' for f in range(5)},
          'comparison':{f'selected__random__{f}__{name}' for f in range(5) for name in ['Logistic_Regression','RBF-SVM','Random_Forest','XGBoost','LightGBM','DNN','1D-CNN']}}
        if proc is None:
            for stage,tasks in needed.items():
                failed={e['stage'] for e in state['errors']}
                if stage not in state['staged'] and stage not in failed and tasks<=completed:
                    command=[sys.executable,'-u',str(ROOT/'scripts/27_report_five_fold.py'),'--stage',stage]
                    with (RUN/'logs'/f'report_{stage}.log').open('w',encoding='utf-8') as handle:
                        proc=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
                    state['current_report']={'stage':stage,'pid':proc.pid,'command':command};print('STAGING',stage,flush=True);break
        if len(state['staged'])==4 and proc is None:
            state['status']='ready_for_visual_review';atomic(state);return
        if primary['status']=='training_complete' and side['status']=='complete' and state['errors'] and proc is None:
            state['status']='report_needs_attention';atomic(state);return
        atomic(state);time.sleep(30)

if __name__=='__main__':main()
