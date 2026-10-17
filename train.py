"""Train one frozen outer fold, with internal validation and resumable prefixes."""
from __future__ import annotations
import argparse
import importlib
import json
import platform
import sys
import time
from pathlib import Path
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
import model as m

ALGORITHMS = ['RBF-SVM', 'Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'DNN', '1D-CNN']
DESIGNS = ['random', 'species'] + [f'homology_{v}' for v in range(10, 100, 10)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', choices=DESIGNS, default='random')
    parser.add_argument('--fold', type=int, choices=range(5), default=0)
    parser.add_argument('--algorithm', choices=ALGORITHMS, default='RBF-SVM')
    parser.add_argument('--features', choices=m.SETS, default='full', help='Non-full groups are predefined ablations (no IFS).')
    parser.add_argument('--output-dir', type=Path, default=Path('runs'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.features != 'full' and args.algorithm != 'RBF-SVM':
        parser.error('Published feature ablations use RBF-SVM only.')
    task = f'{args.algorithm.replace(" ", "_")}__{args.design}__{args.fold}__{args.features}'
    out = args.output_dir.resolve() / task
    if out.exists() and not args.resume:
        raise FileExistsError(f'{out} exists. Inspect it, then use --resume or a new output directory.')
    out.mkdir(parents=True, exist_ok=True)
    config = {'design': args.design, 'fold': args.fold, 'algorithm': args.algorithm,
              'features': args.features, 'seed': m.SEED + args.fold * 100,
              'data_sha256': m.sha(m.ROOT / 'data/processed/archaea_swissprot_ec_master.csv'),
              'features_sha256': m.sha(m.ROOT / 'data/features.npy'),
              'split_sha256': m.sha(m.ROOT / 'data/splits' / f'{args.design}.csv'),
              'code_sha256': {n: m.sha(m.ROOT / n) for n in ['model.py', 'train.py']},
              'python': platform.python_version(),
              'versions': {n: version(n) for n in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib']},
              'ifs_grid': m.GRID, 'refit_with_validation': False}
    if args.algorithm != 'RBF-SVM':
        config['comparator_source_sha256'] = m.sha(m.ROOT / 'archive/scripts/14_mrmr_model_comparison.py')
        config['versions'].update({n: version(n) for n in ['torch', 'lightgbm', 'xgboost']})
    cp = out / 'config.json'
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError('Input/code/environment drift: use a new output directory.')
    m.dump(cp, config)
    if (out / 'complete.json').exists():
        done = json.loads((out / 'complete.json').read_text())
        for name, checksum in done['outputs'].items():
            if m.sha(out / name) != checksum:
                raise ValueError(f'Completed output changed: {name}')
        print(f'Already complete: {out}')
        return
    start = time.monotonic()
    data, x, y = m.load_data()
    tr, va, te = m.split_indices(data, args.design, args.fold)
    predictor = m.predict_scores
    comparison = None
    if args.algorithm != 'RBF-SVM':
        sys.path.insert(0, str(m.ROOT / 'archive/scripts'))
        comparison = importlib.import_module('14_mrmr_model_comparison')
        comparison.OUT = out
        comparison.SEED = config['seed']
        predictor = comparison.predict
        if args.algorithm == '1D-CNN':
            x = comparison.encode_sequences(data.sequence)
    if args.algorithm == '1D-CNN':
        candidates = [np.arange(x.shape[1])]
    elif args.features != 'full':
        candidates = [m.SETS[args.features]]
    else:
        ranking_path = out / 'ranking.npy'
        if ranking_path.exists():
            ranking = np.load(ranking_path)
        else:
            ranking = m.mrmr(x[tr], y[tr])
            np.save(ranking_path, ranking)
        candidates = [ranking[:k] for k in m.GRID]
    choices = []
    for cols in candidates:
        p = out / f'prefix_{len(cols)}'
        p.mkdir(exist_ok=True)
        with threadpool_limits(limits=2 if args.algorithm == 'RBF-SVM' else 4):
            if (p / 'model.joblib').exists():
                pack = joblib.load(p / 'model.joblib')
            else:
                pack = (m.fit_svm(x[tr][:, cols], y[tr], config['seed'], p)
                        if comparison is None else comparison.train_model(args.algorithm, x[tr][:, cols], y[tr], x[va][:, cols], y[va]))
                m.save_model(p / 'model.joblib', pack)
            pv = predictor(pack, x[va][:, cols])
        np.save(p / 'validation_predictions.npy', pv)
        rec = {'n_features': len(cols), 'columns': cols.tolist(),
               'validation_macro_auc': float(roc_auc_score(y[va], pv, average='macro'))}
        m.dump(p / 'validation.json', rec)
        choices.append(rec)
        print(f'{task}: {len(cols)} features; validation AUC={rec["validation_macro_auc"]:.6f}', flush=True)
    best = sorted(choices, key=lambda r: (-r['validation_macro_auc'], r['n_features']))[0]
    cols = np.array(best['columns'])
    selected = out / f'prefix_{len(cols)}'
    pack = joblib.load(selected / 'model.joblib')
    pv = np.load(selected / 'validation_predictions.npy')
    cuts = m.thresholds(y[va], pv)
    scores = predictor(pack, x[te][:, cols])
    result = m.metrics(y[te], scores, cuts)
    pred = data.iloc[te][['accession'] + m.LABELS].copy()
    pred['fold'] = args.fold
    for c in range(7):
        pred[f'score_ec{c+1}'] = scores[:, c]
    if args.algorithm == 'RBF-SVM' and args.features == 'full':
        target = data.is_multifunctional_top_level.astype(int).to_numpy()
        aux = m.fit_svm(x[tr][:, cols], target[tr, None], config['seed'] + 100, out / 'auxiliary_fit')
        auxscore = m.predict_scores(aux, x[te][:, cols])[:, 0]
        pred['score_aux'] = auxscore
        pred['label_aux'] = target[te]
        result['auxiliary_roc_auc'] = float(roc_auc_score(target[te], auxscore))
        result['auxiliary_ap'] = float(average_precision_score(target[te], auxscore))
        m.save_model(out / 'auxiliary.joblib', aux)
        np.save(out / 'validation_auxiliary.npy', m.predict_scores(aux, x[va][:, cols])[:, 0])
    pred.to_csv(out / 'predictions.csv', index=False)
    m.save_model(out / 'model.joblib', pack)
    np.save(out / 'columns.npy', cols)
    np.save(out / 'validation_predictions.npy', pv)
    m.dump(out / 'metrics.json', {'metrics': result, 'thresholds': cuts.tolist(), 'selection': best,
                                 'model': args.algorithm, 'score_type': 'uncalibrated score',
                                 'elapsed_seconds': time.monotonic() - start})
    m.dump(out / 'complete.json', {'status': 'complete', 'outputs': {
        str(p.relative_to(out)).replace('\\', '/'): m.sha(p)
        for p in out.rglob('*') if p.is_file() and p.name != 'complete.json' and not p.name.endswith('.tmp')}})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
