"""Render a QA-only species ROC preview from five completed real folds."""
import importlib.util
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cv_report_preview',ROOT/'scripts/27_report_five_fold.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
loaded,_=m.load_five('lgb__species__');frames=[p for p,r in loaded]
folder=m.RUN/'visual_qa';folder.mkdir(exist_ok=True)
m.style();fig,ax=plt.subplots(figsize=(13,8))
for c in range(7):m.roc(ax,frames,f'EC{c+1} {m.CLASSES[c]}',m.COLORS[c],c)
m.roc_axes(ax);m.save(fig,folder,'species_cv_roc_preview')
print('QA preview only; active result figures not replaced')
