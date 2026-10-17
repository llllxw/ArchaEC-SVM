"""Read-only checks of draft tables, final SVG hashes and local figure links."""
from pathlib import Path
import hashlib
import json
import re
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
draft=ROOT/'manuscript/SCI_results_chinese_draft_v3.md'
body=draft.read_text(encoding='utf-8')

def table(number):
    block=body.split(f'**表{number}　',1)[1].split('表注：',1)[0]
    rows=[[cell.strip() for cell in line.strip().strip('|').split('|')]
          for line in block.splitlines() if line.startswith('|')]
    return rows[2:]

checks=[]
def numeric_rows(actual,expected,name):
    assert len(actual)==len(expected),(name,len(actual),len(expected))
    count=0
    for a,e in zip(actual,expected):
        assert len(a)==len(e),(name,a,e)
        for v,w in zip(a,e):
            expected_text=f'{float(w):.4f}'
            assert v==expected_text,(name,v,expected_text)
            count+=1
    checks.append({'check':name,'numeric_cells':count,'status':'passed'})

comparison=pd.read_csv(ROOT/'results/3.2_feature_selection_model_comparison/algorithm_test_metrics.csv')
numeric_rows([r[1:] for r in table(1)],comparison.iloc[:,1:].values.tolist(),'algorithm table')
hom=pd.read_csv(ROOT/'results/3.3_generalization_ladder/homology_auc_summary.csv')
hom=hom.loc[hom.evaluation.str.startswith('Homology-')]
numeric_rows([r[3:] for r in table(2)],hom[['macro_roc_auc','micro_roc_auc','macro_pr_auc','min_class_roc_auc']].values.tolist(),'homology table')
ablation=pd.read_csv(ROOT/'results/3.4_multifunction_interpretability/feature_ablation_auc_ap.csv').set_index('tag')
ablation=ablation.loc[['full','aac','dpc','phys','aac_dpc','aac_phys','dpc_phys']]
numeric_rows([r[2:] for r in table(3)],ablation[['macro_roc_auc','macro_ap']].values.tolist(),'ablation table')
assert [int(r[1]) for r in table(3)]==ablation.n_features.tolist()
expected_counts=[(9356,1252),(8026,1262),(6654,1263),(4963,1266),(3211,1266),
                 (1939,1254),(1261,1238),(1116,1236),(1116,1236)]
assert [(int(r[1].replace(',','')),int(r[2].replace(',',''))) for r in table(2)]==expected_counts

links=re.findall(r'\]\(([^)]+)\)',body)
for link in links:assert (draft.parent/link).resolve().is_file(),link
assert len(links)==5
manifest=json.loads((ROOT/'docs/final_figures_manifest_20260925.json').read_text(encoding='utf-8'))
for row in manifest['figures']:
    assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
assert len(re.findall(r'^## 3\.[1-5] ',body,flags=re.M))==5
assert all(f'**图{i}　' in body for i in range(1,7))
for stale in ['拟定图','待补最终模型','尚未重跑 SVM','本轮主任务候选模型','30% 同源簇隔离结果未纳入']:
    assert stale not in body,stale
perclass=pd.read_csv(ROOT/'results/3.2_feature_selection_model_comparison/LightGBM_per_class_auc.csv')
for val in perclass.roc_auc:assert f'{val:.4f}' in body
assert abs(perclass.roc_auc.mean()-comparison.set_index('model').loc['LightGBM','ROC-AUC'])<1e-12
report={'status':'verified','draft':draft.relative_to(ROOT).as_posix(),
        'checks':checks,'valid_figure_links':len(links),'approved_hashes_matched':5,
        'results_sections':5,'figure_captions':6,'training_performed':False,
        'scope':'Exact numeric table checks and final figure identity; not a peer-review verdict'}
(ROOT/'docs/manuscript_verification_20260925.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
