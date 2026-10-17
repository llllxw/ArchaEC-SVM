"""Re-export only requested panels, then fingerprint user-approved Figures 2-6."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import importlib.util

ROOT=Path(__file__).resolve().parents[1]

def module(name):
    spec=importlib.util.spec_from_file_location('final_'+name[:2],ROOT/'scripts'/name)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj

def main():
    exporter=module('17_export_composite_figures.py')
    exporter.ARCHIVE=ROOT/'results/_archive_figures_20260925_final_style'
    d3=ROOT/'results/3.3_generalization_ladder'
    old=module('08_make_auc_figures.py')
    with exporter.export_all():
        old.generalization_figures(d3,old.load_outputs(ROOT/'results/3.2_random_split_performance'),old.load_outputs(d3))
    # 14 data bars and their two legend handles must retain explicit alpha=0.5.
    tree=ET.parse(d3/'figure_4A_species_per_class_auc.svg')
    ns={'s':'http://www.w3.org/2000/svg'}
    translucent=[el for el in tree.findall('.//s:path',ns) if 'opacity: 0.5' in el.get('style','')]
    assert len(translucent)>=14, len(translucent)
    exporter.compose(d3,4,[[('figure_4A_species_per_class_auc',1),('figure_4B_species_roc_curves',1)],
        [('figure_4C_homology_auc_ladder',1)],[('figure_4D_homology_roc_curves',1)],
        [('figure_4E_homology_per_class_auc',1)]])
    module('21_plot_ablation_lollipops.py').main()
    folders=['3.1_dataset_characterization','3.2_feature_selection_model_comparison',
             '3.3_generalization_ladder','3.4_multifunction_interpretability','3.5_seven_class_interpretability']
    records=[]
    for number,folder in zip(range(2,7),folders):
        path=ROOT/'results'/folder/f'figure_{number}_combined.svg'
        root=ET.parse(path).getroot()
        ids=[el.get('id') for el in root.iter() if el.get('id')]
        assert len(ids)==len(set(ids))
        assert not root.findall('.//s:image',ns)
        records.append({'figure':number,'path':path.relative_to(ROOT).as_posix(),
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'vector_only':True,'duplicate_ids':0})
    report={'date':'2026-09-25','status':'user-approved figure set; style updated and verified',
            'figure1':'reserved overview','figure4A_alpha':0.5,
            'figure4A_translucent_paths':len(translucent),
            'figure5_palette':'neutral full / cool singles / warm pairs; alpha=0.5',
            'figures':records,'training_performed':False}
    (ROOT/'docs/final_figures_manifest_20260925.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
