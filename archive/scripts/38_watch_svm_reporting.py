"""Stage completed SVM phases once; never retry training or publish automatically."""
from pathlib import Path
import argparse, json, subprocess, sys, time

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'results/svm_migration_20261008'

def main(start='generalization'):
    completed=[];status=RUN/'reporting_status.json'
    if status.exists():raise FileExistsError('Inspect prior reporting status; no automatic restart')
    phases=[('generalization',55),('ablation',85),('interpretability',90),('validate',90)]
    while phases[0][0]!=start:
        completed.append(phases.pop(0)[0])
    main_tasks=[f'svm__{d}__{f}' for d in ['random','species']+[f'homology_{v}' for v in range(10,100,10)] for f in range(5)]
    ablation_tasks=[f'ablation__random__{f}__{tag}' for f in range(5) for tag in ['aac','dpc','phys','aac_dpc','aac_phys','dpc_phys']]
    shap_tasks=[f'shap__random__{f}' for f in range(5)]
    required={'generalization':main_tasks,'ablation':main_tasks+ablation_tasks,
              'interpretability':main_tasks+shap_tasks,'validate':main_tasks+ablation_tasks+shap_tasks}
    while phases:
        state=json.loads((RUN/'run_status.json').read_text(encoding='utf-8'))
        if state['status'] in ['failed','timeout']:
            status.write_text(json.dumps({'status':'stopped','training_status':state['status'],'completed':completed}),encoding='utf-8');return
        phase,count=phases[0]
        ready=all((RUN/'tasks'/task/'complete.json').exists() for task in required[phase])
        if ready and (phase!='validate' or state['status']=='training_complete'):
            record={'status':'running','phase':phase,'completed':completed,'started_at':time.strftime('%Y-%m-%d %H:%M:%S')}
            status.write_text(json.dumps(record),encoding='utf-8')
            print('STAGE',phase,flush=True)
            result=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/36_report_svm_five_fold.py'),'--stage',phase],cwd=ROOT)
            if result.returncode:
                record.update(status='failed',exit_code=result.returncode);status.write_text(json.dumps(record),encoding='utf-8');return
            completed.append(phase);phases.pop(0)
            status.write_text(json.dumps({'status':'waiting' if phases else 'staged_and_validated','completed':completed}),encoding='utf-8')
        else:
            status.write_text(json.dumps({'status':'waiting','next_phase':phase,'training_tasks':state.get('completed_count',0),'completed':completed}),encoding='utf-8')
            time.sleep(30)
    print('READY FOR VISUAL QA; not published',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--from-phase',choices=['generalization','interpretability'],default='generalization')
    main(parser.parse_args().from_phase)
