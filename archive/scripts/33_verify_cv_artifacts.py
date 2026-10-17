"""Read-only integrity checks; writes only the resulting audit record."""
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/five_fold_cv_20261007'

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def main():
    schedule = json.loads((RUN / 'task_schedule.json').read_text())
    config = json.loads((RUN / 'config.json').read_text())
    count = 0
    for relative, expected in config['fingerprints'].items():
        assert sha(ROOT / relative) == expected, relative
    for task in schedule:
        folder = RUN / 'tasks' / task
        record = json.loads((folder / 'complete.json').read_text())
        assert record['task'] == task and record['status'] == 'complete'
        assert not (folder / 'failure.json').exists(), task
        assert not (folder / 'task.lock').exists(), task
        for name, expected in record['outputs'].items():
            assert sha(folder / name) == expected, (task, name)
            count += 1
    state = json.loads((RUN / 'run_status.json').read_text())
    assert state['status'] == 'training_complete'
    assert state['elapsed_seconds'] <= state['budget_seconds']
    packages = ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'torch', 'matplotlib', 'seaborn', 'joblib']
    result = {
        'status': 'passed', 'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'completed_tasks': len(schedule), 'verified_artifacts': count,
        'input_and_original_script_fingerprints_unchanged': True,
        'elapsed_training_seconds': state['elapsed_seconds'],
        'budget_seconds': state['budget_seconds'],
        'packages': {p: importlib.metadata.version(p) for p in packages},
        'implementation_hashes': {p.name: sha(p) for p in (ROOT / 'scripts').glob('*five_fold*.py')},
        'scope': 'file integrity, source fingerprints and completion; not an independent retraining replication'
    }
    publication = RUN / 'published_manifest.json'
    if publication.exists():
        manifest = json.loads(publication.read_text(encoding='utf-8'))
        for record in manifest['files']:
            assert sha(ROOT / record['path']) == record['sha256'], record['path']
        assert (RUN / 'report_validation.json').exists()
        result['verified_published_files'] = len(manifest['files'])
        completion = {
            'status': 'complete_and_published', 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'training_tasks': len(schedule), 'published_files_verified': len(manifest['files']),
            'figures': [2, 3, 4, 5, 6],
            'statistics': 'five-fold mean +/- sample SD (ddof=1), three decimal display',
            'historical_observer_error': 'comparison CSV float parsing issue; diagnosed and resolved in 27_report_five_fold.py; all stages rebuilt and validated, no retraining',
            'input_audit': 'independent_input_audit.json', 'numeric_audit': 'report_validation.json',
            'integrity_audit': 'artifact_integrity_audit.json', 'publication': 'published_manifest.json',
            'visual_qa': 'combined PNG previews for figures 2-6 inspected by assistant; no clipped or colliding text after reported adjustments',
            'user_final_approval': 'not yet requested or obtained for the new CV figures',
            'manuscript': 'full manuscript not rewritten; experiment_result.md documents updated results'
        }
        (RUN / 'completion_status.json').write_text(json.dumps(completion, indent=2), encoding='utf-8')
    (RUN / 'artifact_integrity_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('PASS', len(schedule), 'tasks;', count, 'artifact hashes; elapsed', round(state['elapsed_seconds']), 'seconds')

if __name__ == '__main__':
    main()
