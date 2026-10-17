"""Style-only update: single-row algorithm legend, dual AUC labels, two-column Figure 4."""
from pathlib import Path
import importlib.util
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'results/_archive_figures_20260925_compact_3_4'

def module(name):
    spec=importlib.util.spec_from_file_location('compact_'+name[:2],ROOT/'scripts'/name)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj

def main():
    exporter=module('17_export_composite_figures.py');exporter.ARCHIVE=ARCHIVE
    d3=ROOT/'results/3.2_feature_selection_model_comparison'
    d4=ROOT/'results/3.3_generalization_ladder'
    sources=[d3/'algorithm_test_metrics.csv',d3/'ifs_validation_results.csv',d4/'homology_auc_summary.csv']
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    # The data-driven plotting function generates four panels; export only A here.
    comp=module('14_mrmr_model_comparison.py')
    predictions={}
    for name in comp.NAMES:
        frame=pd.read_csv(d3/'predictions'/f'{name.replace(" ","_")}_test.csv')
        predictions[name]=frame[[f'score_ec{i}' for i in range(1,8)]].to_numpy()
    truth=frame[[f'label_ec{i}' for i in range(1,8)]].to_numpy()
    original_csv=pd.DataFrame.to_csv
    pd.DataFrame.to_csv=lambda *a,**k:None
    try:
        with exporter.export_all():
            wrapped_save=exporter.matplotlib.figure.Figure.savefig
            def save_a(fig,path,*args,**kwargs):
                if Path(path).stem=='figure_3A_algorithm_metrics':return wrapped_save(fig,path,*args,**kwargs)
            exporter.matplotlib.figure.Figure.savefig=save_a
            comp.plots(pd.read_csv(sources[0]),pd.read_csv(sources[1]),truth,predictions,'LightGBM')
    finally:pd.DataFrame.to_csv=original_csv
    exporter.compose(d3,3,[[('figure_3A_algorithm_metrics',1),('figure_3B_algorithm_macro_roc',1)],
                         [('figure_3C_mrmr_ifs',1),('figure_3D_per_class_roc',1)]])
    with exporter.export_all():module('11_summarize_homology_auc.py').plot_ladder(pd.read_csv(sources[2]),d4)
    exporter.compose(d4,4,[[('figure_4A_species_per_class_auc',1),('figure_4B_species_roc_curves',1)],
                         [('figure_4C_homology_auc_ladder',1)],[('figure_4D_homology_roc_curves',1)],
                         [('figure_4E_homology_per_class_auc',1)]])
    ns={'s':'http://www.w3.org/2000/svg'}
    root_c=ET.parse(d4/'figure_4C_homology_auc_ladder.svg').getroot()
    labels=[el.text for el in root_c.findall('.//s:text',ns) if re.fullmatch(r'0\.\d{3}',el.text or '')]
    assert len(labels)==20,labels
    manifest_path=ROOT/'docs/final_figures_manifest_20260925.json'
    backup=ARCHIVE/'final_figures_manifest_20260925.json'
    if not backup.exists():shutil.copy2(manifest_path,backup)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    for row in manifest['figures']:
        if row['figure'] in [3,4]:
            path=ROOT/row['path'];root=ET.parse(path).getroot()
            assert not root.findall('.//s:image',ns)
            ids=[el.get('id') for el in root.iter() if el.get('id')]
            assert len(ids)==len(set(ids))
            row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            row['style_revision']='3A single-row legend' if row['figure']==3 else '4C both AUC series labeled; two columns A/C/E and B/D'
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==v for p,v in before.items())
    report={'status':'verified','figure3A_legend_rows':1,'figure4C_numeric_labels':len(labels),
            'figure4_layout':{'left':['A','C','E'],'right':['B','D']},'source_tables_unchanged':True,'training':False}
    (ROOT/'docs/compact_figures3_4_20260925.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
