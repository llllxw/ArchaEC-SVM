"""Small synthetic CPU installation test; not a scientific evaluation."""
import sys
import json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model as m


def main():
    rng = np.random.default_rng(7)
    sequences = [''.join(rng.choice(list(m.AA), 45)) for _ in range(42)]
    x = m.features(sequences)
    y = np.eye(7, dtype=int)[np.arange(42) % 7]
    pack = m.fit_svm(x[:28], y[:28], 7)
    scores = m.predict_scores(pack, x[28:])
    assert x.shape == (42, 432) and scores.shape == (14, 7)
    assert np.all((scores > 0) & (scores < 1))
    assert np.allclose(x[:, :20].sum(axis=1), 1)
    assert np.allclose(x[:, 20:420].sum(axis=1), 1)
    m.metrics(y[28:], scores, np.full(7, .5))
    print(json.dumps({'status': 'PASS', 'feature_dimensions': 432, 'binary_classifiers': 7,
                      'test_type': 'synthetic installation check; no benchmark performance claim'}))


if __name__ == '__main__':
    main()
