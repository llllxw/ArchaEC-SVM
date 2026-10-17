"""Redraw four Section 3.2 figures from saved scores, without retraining."""
import importlib.util
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('comparison',ROOT/'scripts/14_mrmr_model_comparison.py')
comparison=importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
out=comparison.OUT
summary=json.loads((out/'run_summary.json').read_text(encoding='utf-8'))
table=pd.read_csv(out/'algorithm_test_metrics.csv')
ifs=pd.read_csv(out/'ifs_validation_results.csv')
predictions={}
truth=None
accessions=None
for name in comparison.NAMES:
    frame=pd.read_csv(out/'predictions'/f'{name.replace(" ","_")}_test.csv')
    if accessions is None:
        accessions=frame.accession.tolist()
        truth=frame[[f'label_ec{i}' for i in range(1,8)]].to_numpy()
    assert frame.accession.tolist()==accessions
    predictions[name]=frame[[f'score_ec{i}' for i in range(1,8)]].to_numpy()
comparison.plots(table,ifs,truth,predictions,'LightGBM')
print('Recreated four PNG/PDF pairs from saved scores; no model training.')
