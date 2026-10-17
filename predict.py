"""Predict seven EC labels from a CSV (accession,sequence) using a trusted SVM run."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import model as m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--input-csv', type=Path, required=True)
    p.add_argument('--output-csv', type=Path, required=True)
    args = p.parse_args()
    if args.output_csv.exists():
        raise FileExistsError(args.output_csv)
    frame = pd.read_csv(args.input_csv)
    if not {'accession', 'sequence'}.issubset(frame.columns) or not frame.accession.is_unique:
        raise ValueError('Input requires unique accession and sequence columns.')
    if frame.empty:
        raise ValueError('Input is empty.')
    # joblib is executable pickle: load only checkpoints you trust.
    pack = joblib.load(args.run_dir / 'model.joblib')
    if pack['name'] != 'RBF-SVM':
        raise ValueError('This inference entry point is for the final RBF-SVM model.')
    cols = np.load(args.run_dir / 'columns.npy')
    rec = json.loads((args.run_dir / 'metrics.json').read_text())
    scores = m.predict_scores(pack, m.features(frame.sequence)[:, cols])
    result = frame[['accession']].copy()
    for c in range(7):
        result[f'score_ec{c+1}'] = scores[:, c]
        result[f'predicted_ec{c+1}'] = (scores[:, c] >= rec['thresholds'][c]).astype(int)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print('Saved uncalibrated scores and validation-thresholded labels:', args.output_csv)


if __name__ == '__main__':
    main()
