"""Portable SVM feature/model implementation matching the frozen experiment.

Scores are sigmoid-transformed margins, NOT calibrated probabilities.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                            matthews_corrcoef, precision_score, recall_score,
                            roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
AA = 'ACDEFGHIKLMNPQRSTVWY'
LABELS = [f'label_ec{i}' for i in range(1, 8)]
CLASSES = ['Oxidoreductases', 'Transferases', 'Hydrolases', 'Lyases',
           'Isomerases', 'Ligases', 'Translocases']
PHYS = ['log_length', 'sequence_entropy', 'longest_hydrophobic_run',
        'hydrophobic_fraction', 'charged_fraction', 'polar_fraction',
        'aromatic_fraction', 'small_fraction', 'glycine_fraction',
        'proline_fraction', 'cysteine_fraction', 'histidine_fraction']
FEATURES = [f'AAC_{a}' for a in AA] + [f'DPC_{a}{b}' for a in AA for b in AA] + PHYS
GROUPS = [set(s) for s in ['AVILMFWY', 'DEKR', 'STNQ', 'FWY', 'AGST', 'G', 'P', 'C', 'H']]
GRID = [16, 32, 64, 128, 256, 432]
SETS = {'full': np.arange(432), 'aac': np.arange(20), 'dpc': np.arange(20, 420),
        'phys': np.arange(420, 432), 'aac_dpc': np.arange(420),
        'aac_phys': np.r_[0:20, 420:432], 'dpc_phys': np.arange(20, 432)}
SEED = 20261007


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def save_model(path, value):
    path = Path(path)
    temp = path.with_suffix('.joblib.tmp')
    joblib.dump(value, temp)
    temp.replace(path)


def sequence_features(sequence):
    if not isinstance(sequence, str) or not sequence or sequence != sequence.upper():
        raise ValueError('Sequences must be nonempty uppercase strings.')
    if not set(sequence).issubset(set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')):
        raise ValueError('Sequences must contain amino-acid letters only (no gaps or stops).')
    lookup = {a: i for i, a in enumerate(AA)}
    aac = np.zeros(20, dtype=np.float32)
    dpc = np.zeros((20, 20), dtype=np.float32)
    residues = [a for a in sequence if a in lookup]
    if not residues:
        raise ValueError('Sequence has no canonical amino acids.')
    for a in residues:
        aac[lookup[a]] += 1
    aac /= len(residues)
    pairs = 0
    for a, b in zip(sequence, sequence[1:]):
        if a in lookup and b in lookup:
            dpc[lookup[a], lookup[b]] += 1
            pairs += 1
    if pairs:
        dpc /= pairs
    counts = np.array([residues.count(a) for a in AA], dtype=float)
    p = counts[counts > 0] / len(residues)
    longest = current = 0
    for a in sequence:
        current = current + 1 if a in GROUPS[0] else 0
        longest = max(longest, current)
    physical = [np.log1p(len(sequence)), float(-(p * np.log2(p)).sum()), longest / len(residues)]
    physical.extend(sum(a in g for a in residues) / len(residues) for g in GROUPS)
    return np.concatenate([aac, dpc.ravel(), np.asarray(physical, dtype=np.float32)])


def features(sequences):
    return np.vstack([sequence_features(s) for s in sequences])


def load_data():
    data = pd.read_csv(ROOT / 'data/processed/archaea_swissprot_ec_master.csv', dtype={'taxonomy_id': str})
    if not data.accession.is_unique:
        raise ValueError('Duplicate accession IDs.')
    ids = (ROOT / 'data/features.accessions.txt').read_text(encoding='utf-8').splitlines()
    if ids != data.accession.tolist():
        raise ValueError('Feature row order differs from master accession order.')
    x = np.load(ROOT / 'data/features.npy')
    if x.shape != (len(data), 432) or not np.isfinite(x).all():
        raise ValueError('Invalid feature matrix.')
    return data, x, data[LABELS].to_numpy(int)


def split_indices(data, design, fold):
    frame = pd.read_csv(ROOT / 'data/splits' / f'{design}.csv', dtype={'group': str})
    if frame.accession.tolist() != data.accession.tolist():
        raise ValueError('Split accession order differs from master.')
    role = frame[f'role_fold_{fold}'].to_numpy()
    if set(role) != {'train', 'validation', 'test'}:
        raise ValueError('Invalid split roles.')
    indices = [np.flatnonzero(role == name) for name in ['train', 'validation', 'test']]
    for column in ['group', 'sequence_sha256']:
        values = frame[column] if column == 'group' else data[column]
        sets = [set(values.iloc[ix]) for ix in indices]
        if any(sets[a] & sets[b] for a, b in [(0, 1), (0, 2), (1, 2)]):
            raise ValueError(f'Cross-partition overlap: {column}')
    return indices


def discrete_mi(a, b, bins_b):
    counts = np.bincount(a.astype(np.int64) * bins_b + b, minlength=10 * bins_b).reshape(10, bins_b)
    joint = counts / counts.sum()
    expected = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))


def mrmr(x, y):
    """Training-only ten-bin quantile mRMR-MID; deterministic tie handling."""
    edges = [np.unique(np.quantile(x[:, j], np.arange(1, 10) / 10)) for j in range(x.shape[1])]
    z = np.column_stack([np.searchsorted(e, x[:, j], side='left') for j, e in enumerate(edges)])
    relevance = np.array([[discrete_mi(z[:, j], y[:, c], 2) for c in range(7)] for j in range(x.shape[1])])
    mean = relevance.mean(axis=1)
    redundancy_sum = np.zeros(x.shape[1])
    chosen = []
    for step in range(x.shape[1]):
        score = mean - redundancy_sum / max(step, 1)
        score[chosen] = -np.inf
        j = int(np.argmax(score))
        chosen.append(j)
        for other in range(x.shape[1]):
            if other not in chosen:
                redundancy_sum[other] += discrete_mi(z[:, other], z[:, j], 10)
    return np.array(chosen)


def fit_svm(x, y, seed, checkpoint_dir=None):
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x).astype(np.float32)
    models = []
    for c in range(y.shape[1]):
        target = None if checkpoint_dir is None else Path(checkpoint_dir) / f'ec{c+1}.joblib'
        if target is not None and target.exists():
            model = joblib.load(target)
        else:
            if len(np.unique(y[:, c])) != 2:
                raise ValueError(f'Class {c+1} is not estimable.')
            model = SVC(C=2, gamma='scale', class_weight='balanced', cache_size=512, random_state=seed)
            model.fit(scaled, y[:, c])
            if target is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                save_model(target, model)
        models.append(model)
    return {'scaler': scaler, 'models': models, 'name': 'RBF-SVM'}


def margins(pack, x):
    scaled = pack['scaler'].transform(x).astype(np.float32)
    return np.column_stack([m.decision_function(scaled) for m in pack['models']])


def predict_scores(pack, x):
    return expit(margins(pack, x))


def thresholds(y, scores):
    return np.array([max(np.linspace(.01, .99, 99), key=lambda t: f1_score(y[:, j], scores[:, j] >= t, zero_division=0)) for j in range(7)])


def metrics(y, scores, cuts):
    if y.shape != scores.shape or y.shape[1] != 7 or not np.isfinite(scores).all():
        raise ValueError('Invalid seven-label truth/score matrices.')
    if any(len(np.unique(y[:, c])) != 2 for c in range(7)):
        raise ValueError('All seven labels require positives and negatives.')
    pred = scores >= cuts
    return {'ROC-AUC': float(roc_auc_score(y, scores, average='macro')),
            'AP': float(average_precision_score(y, scores, average='macro')),
            'Accuracy': float(accuracy_score(y, pred)),
            'Precision': float(precision_score(y, pred, average='macro', zero_division=0)),
            'Recall': float(recall_score(y, pred, average='macro', zero_division=0)),
            'F1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'Specificity': float(np.mean([np.mean(~pred[y[:, j] == 0, j]) for j in range(7)])),
            'MCC': float(np.mean([matthews_corrcoef(y[:, j], pred[:, j]) for j in range(7)])),
            'micro_roc_auc': float(roc_auc_score(y, scores, average='micro'))}
