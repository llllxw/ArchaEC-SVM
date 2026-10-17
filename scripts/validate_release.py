"""Read-only audit of release hashes, folds, features, and stored predictions."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model as m


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-features', action='store_true', help='Recompute the complete released feature matrix.')
    args = parser.parse_args()
    manifest = json.loads((m.ROOT / 'docs/source_file_manifest.json').read_text(encoding='utf-8'))
    for record in manifest['files']:
        require(m.sha(m.ROOT / record['release_path']) == record['sha256'], 'Changed copied source: ' + record['release_path'])
    data, x, y = m.load_data()
    require(len(data) == 11425 and data.sequence.nunique() == 10854, 'Dataset size mismatch')
    require(data.taxonomy_id.nunique() == 204, 'Taxon count mismatch')
    require(np.array_equal(y.sum(axis=0), [1215, 4030, 2301, 1330, 734, 1807, 271]), 'Label counts differ')
    require(int((y.sum(axis=1) > 1).sum()) == 262, 'Multilabel count differs')
    require(np.array_equal(y.sum(axis=1), data.n_top_classes), 'Label count metadata mismatch')
    for row in data.itertuples():
        require(hashlib.sha256(row.sequence.encode()).hexdigest() == row.sequence_sha256, 'Sequence hash mismatch')
    indices = np.arange(len(data)) if args.all_features else np.linspace(0, len(data)-1, 100, dtype=int)
    require(np.array_equal(m.features(data.sequence.iloc[indices]), x[indices]), 'Feature implementation differs from frozen cache')
    designs = ['random', 'species'] + [f'homology_{v}' for v in range(10, 100, 10)]
    checks = 0
    summary = pd.read_csv(m.ROOT / 'results/3.3_generalization_ladder/homology_auc_summary_numeric.csv').set_index('evaluation')
    for design in designs:
        split = pd.read_csv(m.ROOT / 'data/splits' / f'{design}.csv')
        test_roles = split[[f'role_fold_{f}' for f in range(5)]].eq('test').sum(axis=1)
        require(test_roles.eq(1).all(), 'Not exactly one outer test assignment')
        oof = pd.read_csv(m.ROOT / f'results/3.3_generalization_ladder/oof_{design}.csv', float_precision='round_trip')
        require(oof.accession.is_unique and set(oof.accession) == set(data.accession), 'OOF coverage mismatch')
        fold_metrics = []
        for f in range(5):
            tr, va, te = m.split_indices(data, design, f)
            if design == 'species':
                sets = [set(data.taxonomy_id.iloc[ix]) for ix in [tr, va, te]]
                require(not(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), 'Species leakage')
            pred = oof[oof.fold.eq(f)].set_index('accession')
            expected_ids = data.accession.iloc[te].tolist()
            require(set(pred.index) == set(expected_ids), 'OOF/fold IDs mismatch')
            pred = pred.loc[expected_ids]
            require(np.array_equal(pred[m.LABELS].to_numpy(), y[te]), 'OOF labels differ from source')
            rec = json.loads((m.ROOT / f'results/evidence/svm__{design}__{f}/metrics.json').read_text())
            scores = pred[[f'score_ec{i}' for i in range(1, 8)]].to_numpy()
            metric = m.metrics(y[te], scores, np.array(rec['thresholds']))
            for name, value in metric.items():
                require(abs(value - rec['metrics'][name]) < 1e-11, f'Metric differs: {design}/{f}/{name}')
            fold_metrics.append(metric)
            checks += 1
        for name in ['ROC-AUC', 'AP', 'micro_roc_auc']:
            values = np.array([row[name] for row in fold_metrics])
            require(abs(values.mean() - summary.loc[design, name + '_mean']) < 1e-11, 'Summary mean differs')
            require(abs(values.std(ddof=1) - summary.loc[design, name + '_sd']) < 1e-11, 'Summary SD differs')
    require(m.sha(m.ROOT / 'data/splits/homology_10.csv') == m.sha(m.ROOT / 'data/splits/homology_20.csv'), '10/20 split equality changed')
    secondary_checks = 0
    for section, summary_name, key, prefix in [
        ('3.2_feature_selection_model_comparison', 'algorithm_test_metrics_numeric.csv', 'model', 'oof_'),
        ('3.4_multifunction_interpretability', 'feature_ablation_auc_ap_numeric.csv', 'tag', 'oof_')]:
        folder = m.ROOT / 'results' / section
        table = pd.read_csv(folder / summary_name)
        for _, row in table.iterrows():
            frame = pd.read_csv(folder / (prefix + row[key].replace(' ', '_') + '.csv'), float_precision='round_trip')
            require(frame.accession.is_unique and set(frame.accession) == set(data.accession), 'Secondary OOF coverage mismatch')
            values = []
            for f in range(5):
                sub = frame[frame.fold.eq(f)].set_index('accession')
                te = m.split_indices(data, 'random', f)[2]
                ids = data.accession.iloc[te].tolist()
                require(set(sub.index) == set(ids), 'Secondary OOF partition mismatch')
                sub = sub.loc[ids]
                require(np.array_equal(sub[m.LABELS].to_numpy(), y[te]), 'Secondary OOF labels mismatch')
                scores = sub[[f'score_ec{i}' for i in range(1, 8)]].to_numpy()
                values.append([roc_auc_score(y[te], scores, average='macro'), average_precision_score(y[te], scores, average='macro')])
                secondary_checks += 1
            for i, metric in enumerate(['ROC-AUC', 'AP']):
                array = np.asarray(values)[:, i]
                require(abs(array.mean() - row[metric+'_mean']) < 1e-11, 'Secondary mean mismatch')
                require(abs(array.std(ddof=1) - row[metric+'_sd']) < 1e-11, 'Secondary SD mismatch')
    print(json.dumps({'status': 'PASS', 'source_files_verified': len(manifest['files']),
                      'accessions': len(data), 'unique_sequences': data.sequence.nunique(),
                      'feature_rows_recomputed': len(indices), 'fold_predictions_recomputed': checks,
                      'comparison_and_ablation_fold_auc_ap_checks': secondary_checks,
                      'all_data_read_only': True}, indent=2))


if __name__ == '__main__':
    main()
