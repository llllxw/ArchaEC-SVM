"""Widen Figure 3, horizontal metric labels, lower-right IFS legend; no training."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import shutil
import xml.etree.ElementTree as ET
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/3.2_feature_selection_model_comparison'
ARCHIVE=ROOT/'results/_archive_figures_20260925_figure3_wide'

def module(name):
    spec=importlib.util.spec_from_file_location('wide_'+name[:2],ROOT/'scripts'/name)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj

def main():
    exporter=module('17_export_composite_figures.py');exporter.ARCHIVE=ARCHIVE
    comp=module('14_mrmr_model_comparison.py')
    evidence=list(OUT.glob('*.csv'))+list((OUT/'predictions').glob('*.csv'))
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}
    original_pdf=ARCHIVE/'3.2_feature_selection_model_comparison/figure_3_combined.pdf'
    if original_pdf.exists():
        with exporter.fitz.open(original_pdf) as old_pdf:
            oldbox=[0,0,old_pdf[0].rect.width,old_pdf[0].rect.height]
    elif (OUT/'figure_3_combined.svg').exists():
        oldroot=ET.parse(OUT/'figure_3_combined.svg').getroot()
        oldbox=list(map(float,oldroot.get('viewBox').split()))
    else:
        with exporter.fitz.open(OUT/'figure_3_combined.pdf') as old_pdf:
            oldbox=[0,0,old_pdf[0].rect.width,old_pdf[0].rect.height]
    table=pd.read_csv(OUT/'algorithm_test_metrics.csv')
    ifs=pd.read_csv(OUT/'ifs_validation_results.csv')
    predictions={};truth=None
    for name in comp.NAMES:
        frame=pd.read_csv(OUT/'predictions'/f'{name.replace(" ","_")}_test.csv')
        y=frame[[f'label_ec{i}' for i in range(1,8)]].to_numpy()
        if truth is not None:assert (truth==y).all()
        truth=y
        predictions[name]=frame[[f'score_ec{i}' for i in range(1,8)]].to_numpy()
    # plots() normally emits a derived per-class table; suppress this write in style-only mode.
    original_to_csv=pd.DataFrame.to_csv
    def no_table_write(*args,**kwargs):return None
    pd.DataFrame.to_csv=no_table_write
    try:
        with exporter.export_all():comp.plots(table,ifs,truth,predictions,'LightGBM')
    finally:pd.DataFrame.to_csv=original_to_csv
    exporter.compose(OUT,3,[[('figure_3A_algorithm_metrics',1),('figure_3B_algorithm_macro_roc',1)],
                           [('figure_3C_mrmr_ifs',1),('figure_3D_per_class_roc',1)]])
    for p,digest in before.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,p
    root=ET.parse(OUT/'figure_3_combined.svg').getroot()
    ns={'s':'http://www.w3.org/2000/svg'}
    assert not root.findall('.//s:image',ns)
    ids=[el.get('id') for el in root.iter() if el.get('id')]
    assert len(ids)==len(set(ids))
    panel=ET.parse(OUT/'figure_3C_mrmr_ifs.svg').getroot()
    text=' '.join(el.text or '' for el in panel.findall('.//s:text',ns))
    for name in comp.NAMES[:-1]:assert name in text,name
    assert 'Model [features, peak AUC]' in text
    panel_a=ET.parse(OUT/'figure_3A_algorithm_metrics.svg').getroot()
    for el in panel_a.findall('.//s:text',ns):
        if el.text in ['ROC-AUC','AP','Accuracy','Precision','Recall','F1','Specificity','MCC']:
            rotation=re.search(r'rotate\(([-\d.]+)',el.get('transform',''))
            assert rotation is None or float(rotation.group(1))==0,el.attrib
    newbox=list(map(float,root.get('viewBox').split()))
    manifest_path=ROOT/'docs/final_figures_manifest_20260925.json'
    backup=ARCHIVE/'final_figures_manifest_20260925.json'
    if not backup.exists():shutil.copy2(manifest_path,backup)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    for row in manifest['figures']:
        if row['figure']==3:
            row['sha256']=hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()
            row['style_revision']='wide canvas; horizontal A labels; lower-right C legend'
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    report={'status':'verified','evidence_files_unchanged':len(before),'training':False,
            'previous_aspect_ratio':oldbox[2]/oldbox[3],'new_aspect_ratio':newbox[2]/newbox[3],
            'ifs_curves':6,'ifs_legend':'lower right; model and [features, peak AUC]',
            'metric_label_rotation':0,'panel_size_inches':[12.5,6]}
    (ROOT/'docs/figure3_wide_revision_20260925.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
