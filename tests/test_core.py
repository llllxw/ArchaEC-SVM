import ast
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
import model as m


def source_functions(filename, names, namespace):
    tree = ast.parse((m.ROOT / 'archive/scripts' / filename).read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), namespace)
    return namespace


class CoreTests(unittest.TestCase):
    def test_features_match_original(self):
        comp = source_functions('05_run_aac_dpc_baseline.py', ['features_for_sequence'],
                                {'np': np, 'AA_INDEX': {a: i for i, a in enumerate(m.AA)}})
        groups = dict(zip(['hydrophobic', 'charged', 'polar', 'aromatic', 'small', 'glycine', 'proline', 'cysteine', 'histidine'], m.GROUPS))
        phys = source_functions('07_run_lightgbm_auc_model.py', ['physicochemical_features'],
                                {'np': np, 'pd': pd, 'RESIDUE_GROUPS': groups})
        sequences = ['ACDEFGHIKLMNPQRSTVWY', 'AXCA', 'GGGG', 'W', 'MABZILK']
        expected = np.hstack([np.vstack([comp['features_for_sequence'](s) for s in sequences]),
                              phys['physicochemical_features'](pd.Series(sequences))])
        np.testing.assert_array_equal(m.features(sequences), expected)

    def test_mrmr_matches_original(self):
        rng = np.random.default_rng(42)
        x = rng.integers(0, 4, (50, 12)).astype(float)
        y = rng.integers(0, 2, (50, 7))
        with patch.object(pd.DataFrame, 'to_csv', return_value=None):
            ns = source_functions('14_mrmr_model_comparison.py', ['discrete_mi', 'mrmr'],
                                  {'np': np, 'pd': pd, 'OUT': Path('unused-test-output'),
                                   'FEATURES': m.FEATURES, 'save_json': lambda *a: None})
            np.testing.assert_array_equal(m.mrmr(x, y), ns['mrmr'](x, y))

    def test_invalid_sequences(self):
        for s in ['', 'acde', 'AC-D', 'AC*D', 'XXX']:
            with self.assertRaises(ValueError):
                m.sequence_features(s)

    def test_invalid_auc_population(self):
        with self.assertRaises(ValueError):
            m.metrics(np.zeros((10, 7)), np.full((10, 7), .5), np.full(7, .5))

    def test_thresholds_match_original(self):
        from sklearn.metrics import f1_score
        ns = source_functions('14_mrmr_model_comparison.py', ['thresholds'], {'np': np, 'f1_score': f1_score})
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, (80, 7))
        scores = rng.random((80, 7))
        np.testing.assert_array_equal(m.thresholds(y, scores), ns['thresholds'](y, scores))

    def test_reordered_split_fails(self):
        data, _, _ = m.load_data()
        with self.assertRaises(ValueError):
            m.split_indices(data.iloc[::-1].reset_index(drop=True), 'random', 0)


if __name__ == '__main__':
    unittest.main()
