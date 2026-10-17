"""Sequential five-fold runner with a cumulative time limit and checkpoints."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
import model as m
from train import ALGORITHMS, DESIGNS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design', choices=DESIGNS, default='random')
    p.add_argument('--algorithm', choices=ALGORITHMS, default='RBF-SVM')
    p.add_argument('--features', choices=m.SETS, default='full')
    p.add_argument('--output-dir', type=Path, default=Path('runs'))
    p.add_argument('--hours', type=float, default=6)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.hours <= 0:
        p.error('--hours must be positive')
    name = f'{args.algorithm.replace(" ", "_")}__{args.design}__{args.features}'
    status_path = args.output_dir / f'{name}__status.json'
    prior = json.loads(status_path.read_text()) if status_path.exists() else {}
    if prior and not args.resume:
        raise FileExistsError('Existing run; inspect then use --resume.')
    used = prior.get('elapsed_seconds', 0.0)
    start = time.monotonic()
    budget = args.hours * 3600
    state = {'status': 'running', 'elapsed_seconds': used, 'budget_seconds': budget}
    m.dump(status_path, state)
    for f in range(5):
        remaining = budget - used - (time.monotonic() - start)
        if remaining <= 0:
            state['status'] = 'time_limit'
            break
        command = [sys.executable, str(m.ROOT / 'train.py'), '--design', args.design,
                   '--fold', str(f), '--algorithm', args.algorithm, '--features', args.features,
                   '--output-dir', str(args.output_dir.resolve())]
        if args.resume:
            command.append('--resume')
        try:
            subprocess.run(command, check=True, timeout=remaining)
        except subprocess.TimeoutExpired:
            state['status'] = 'time_limit'
            break
        except subprocess.CalledProcessError:
            state['status'] = 'failed'
            state['elapsed_seconds'] = used + time.monotonic() - start
            m.dump(status_path, state)
            raise
        finally:
            state['elapsed_seconds'] = used + time.monotonic() - start
            m.dump(status_path, state)
    else:
        rows = []
        for f in range(5):
            folder = args.output_dir / f'{args.algorithm.replace(" ", "_")}__{args.design}__{f}__{args.features}'
            rec = json.loads((folder / 'metrics.json').read_text())
            rows.append({'fold': f, **rec['metrics']})
        raw = pd.DataFrame(rows)
        raw.to_csv(args.output_dir / f'{name}__folds.csv', index=False)
        summary = raw.drop(columns='fold').agg(['mean', 'std']).T
        summary.to_csv(args.output_dir / f'{name}__summary_numeric.csv')
        state['status'] = 'complete'
    m.dump(status_path, state)
    print(json.dumps(state, indent=2))
    if state['status'] != 'complete':
        sys.exit(2)


if __name__ == '__main__':
    main()
