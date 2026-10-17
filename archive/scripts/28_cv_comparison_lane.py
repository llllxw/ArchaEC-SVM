"""Second CPU training lane; same tasks/configuration and cumulative deadline.

Task-level exclusive leases prevent two processes fitting the same task. This
lane neither resets the main budget nor retries failures. No data are uploaded.
"""
import importlib.util, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cv_driver',ROOT/'scripts/26_five_fold_experiments.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
RUN=driver.RUN

def main():
    statefile=RUN/'comparison_lane_status.json'
    if statefile.exists():raise FileExistsError('Existing second lane: inspect; no automatic retry')
    state={'status':'running','pid':os.getpid(),'completed':[],'threads_per_worker':4}
    def save():driver.dump(statefile,state)
    save()
    try:
        jobs=[t for t in driver.tasklist() if t.split('__')[0] in ['rank','compare','selected']]
        state['total_tasks']=len(jobs)
        for task in jobs:
            if (driver.taskdir(task)/'complete.json').exists():state['completed'].append(task);continue
            primary=json.loads((RUN/'run_status.json').read_text())
            if primary['status'] in ['failed','timeout']:raise RuntimeError('Primary lane stopped; inspect before continuing')
            remaining=primary['budget_seconds']-primary['elapsed_seconds']
            if remaining<=0:raise TimeoutError('Shared budget exhausted')
            state['current_task']=task;save()
            with (RUN/'logs'/f'comparison_lane_{task}.log').open('w',encoding='utf-8') as f:
                proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/26_five_fold_experiments.py'),'--worker',task],stdout=f,stderr=subprocess.STDOUT,
                    env={**os.environ,'PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
                state['worker_pid']=proc.pid;save();start=time.monotonic()
                while proc.poll() is None:
                    time.sleep(10)
                    state['current_task_elapsed']=time.monotonic()-start;save()
                    if time.monotonic()-start>=remaining:
                        print('HARD TIMEOUT: stop comparison worker',proc.pid,flush=True)
                        subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
                        raise TimeoutError('Shared six-hour limit reached; checkpoints retained')
                if proc.returncode:raise RuntimeError(f'{task} failed with {proc.returncode}; no automatic retry')
            state['completed'].append(task);save();print('COMPARISON',len(state['completed']),len(jobs),task,flush=True)
        state['status']='complete';save()
    except Exception as exc:
        state['status']='timeout' if isinstance(exc,TimeoutError) else 'failed';state['error']=str(exc);save();raise

if __name__=='__main__':main()
