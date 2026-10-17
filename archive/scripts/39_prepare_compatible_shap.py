"""Archive provisional SHAP outputs and switch only project-local dependencies.

This repairs declared dependency compatibility; it does not change SVM models,
data, folds, seeds, sampling or explanation settings. Resume retains the budget.
"""
from pathlib import Path
import hashlib, json, shutil, time

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'results/svm_migration_20261008'
ARCHIVE=RUN/'legacy_shap052_numpy126'

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def move(src,dst):
    assert src.resolve().is_relative_to(ROOT.resolve()) and dst.resolve().is_relative_to(ROOT.resolve())
    assert src.exists() and not dst.exists(),(src,dst)
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(src),str(dst))

def main():
    state=json.loads((RUN/'run_status.json').read_text(encoding='utf-8'))
    reporting=json.loads((RUN/'reporting_status.json').read_text(encoding='utf-8'))
    assert state['status']=='training_complete' and not state['active'] and not state['errors']
    assert reporting['status']=='staged_and_validated'
    assert not ARCHIVE.exists();ARCHIVE.mkdir()
    unchanged={str(p.parent.name):digest(p) for p in (RUN/'tasks').glob('*/complete.json') if not p.parent.name.startswith('shap__')}
    assert len(unchanged)==85
    shutil.copy2(RUN/'run_status.json',ARCHIVE/'original_run_status.json')
    for f in range(5):move(RUN/'tasks'/f'shap__random__{f}',ARCHIVE/'tasks'/f'shap__random__{f}')
    move(RUN/'staged_results/3.5_seven_class_interpretability',ARCHIVE/'staged_results/3.5_seven_class_interpretability')
    for name in ['reporting_status.json','report_validation.json','artifact_integrity_audit.json',
                 'independent_input_audit.json','execution_environment.json','statistical_validation_notes.md','experiment_result.md']:
        if (RUN/name).exists():move(RUN/name,ARCHIVE/'reports'/name)
    for f in range(5):move(RUN/'logs'/f'shap__random__{f}.log',ARCHIVE/'logs'/f'shap__random__{f}.log')
    move(ROOT/'.svm_dependencies',ROOT/'.svm_dependencies_shap052_numpy126')
    move(ROOT/'.svm_dependencies_compatible',ROOT/'.svm_dependencies')
    record={'reason':'SHAP 0.52.0 declares numpy>=2, but environment is numpy 1.26.4',
        'repair':'project-local SHAP 0.49.1; all declared base requirements and KernelSHAP preflight verified',
        'status':'prepared for explanation-only resume','time':time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_tasks_unchanged':unchanged,'rerun_tasks':[f'shap__random__{f}' for f in range(5)],
        'parameters_changed':False,'budget_reset':False,'previous_elapsed_seconds':state['elapsed_seconds'],
        'command':'E:/anaconda3/python.exe -u scripts/35_svm_five_fold_experiments.py --hours 6 --resume',
        'archive':str(ARCHIVE.relative_to(ROOT))}
    (RUN/'dependency_compatibility_repair.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('PREPARED: 85 tasks unchanged; 5 provisional explanations archived; budget retained',flush=True)

if __name__=='__main__':main()
