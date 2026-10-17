"""Replot saved evidence and compose genuine vector SVG/PDF; no training.

2026-09-24 layout follows the user's on-disk numbering (Figure 4 = generalization).
Historical graphics are recoverably archived before replacement.
"""
from pathlib import Path
import importlib.util
import json
import re
import shutil
import xml.etree.ElementTree as ET
from contextlib import contextmanager
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz
import textwrap
from matplotlib.text import Text
from plot_style import apply_arial_style

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
ARCHIVE=RESULTS/'_archive_figures_20260924_before_full_ablation'
ORIGINAL_SAVE=matplotlib.figure.Figure.savefig
NS='http://www.w3.org/2000/svg'
ET.register_namespace('',NS)
ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
LOG=[]
FONT_AUDIT=[]

def module(filename):
    spec=importlib.util.spec_from_file_location('replot_'+filename[:2],ROOT/'scripts'/filename)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def archive(path,move=False):
    if not path.exists(): return
    target=ARCHIVE/path.relative_to(RESULTS)
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists(): shutil.copy2(path,target)
    if move:
        # Exact validated file only; snapshot already saved, no recursive deletion.
        assert path.resolve().is_relative_to(RESULTS.resolve())
        path.unlink()

@contextmanager
def export_all():
    seen=set()
    def save(fig,path,*args,**kwargs):
        path=Path(path)
        stem=path.with_suffix('')
        if stem in seen: return
        seen.add(stem)
        fig.canvas.draw()
        checked=0
        for item in fig.findobj(match=Text):
            before=item.get_fontsize()
            item.set_fontsize(before*2)
            assert item.get_fontsize()==before*2
            checked+=1
        FONT_AUDIT.append({'panel':stem.name,'text_objects_doubled':checked})
        for ax in fig.axes:
            if ax.get_title(): ax.set_title(textwrap.fill(ax.get_title(),width=48),fontsize=ax.title.get_fontsize())
        fig.tight_layout(pad=1.1)
        for ext in ['.svg','.pdf','.png']:
            target=stem.with_suffix(ext)
            archive(target)
            # Heatmaps and colorbars must remain vector in SVG/PDF.
            for ax in fig.axes:
                for artist in ax.collections: artist.set_rasterized(False)
            ORIGINAL_SAVE(fig,target,dpi=300,bbox_inches='tight')
        LOG.append(str(stem.relative_to(ROOT)))
    matplotlib.figure.Figure.savefig=save
    try: yield
    finally: matplotlib.figure.Figure.savefig=ORIGINAL_SAVE

def compose(folder,number,rows):
    """rows: lists of (stem, relative cell width), one letter per input panel.

    SVG preserves all original paths and editable Arial text. PDF uses the same
    placement with embedded vector pages. Prefix all IDs to avoid clip collisions.
    """
    width=1152.0; margin=16.; gap=16.; y=margin
    layout=[]; letter=0
    if number==4:
        # Two compact columns; keep the scientific panel letters unchanged.
        stems=[stem for row in rows for stem,_ in row]
        assert len(stems)==5
        cell=(width-2*margin-gap)/2
        for column,indices in enumerate([[0,2,4],[1,3]]):
            column_y=margin
            for index in indices:
                stem=stems[index]
                svg=ET.parse(folder/(stem+'.svg')).getroot()
                vb=list(map(float,svg.attrib['viewBox'].split()))
                h=vb[3]*cell/vb[2]
                layout.append((stem,svg,margin+column*(cell+gap),column_y+24,cell,h,chr(65+index)))
                column_y+=h+24+gap
            y=max(y,column_y)
    for row in ([] if number==4 else rows):
        available=width-2*margin-gap*(len(row)-1)
        total=sum(weight for _,weight in row)
        x=margin; heights=[]
        for stem,weight in row:
            cell=available*weight/total
            svg=ET.parse(folder/(stem+'.svg')).getroot()
            vb=list(map(float,svg.attrib['viewBox'].split()))
            scale=cell/vb[2]
            h=vb[3]*scale
            layout.append((stem,svg,x,y+24,cell,h,chr(65+letter)))
            heights.append(h); x+=cell+gap; letter+=1
        y+=max(heights)+24+gap
    height=y
    root=ET.Element(f'{{{NS}}}svg',{'width':f'{width}pt','height':f'{height}pt',
                      'viewBox':f'0 0 {width} {height}'})
    ET.SubElement(root,f'{{{NS}}}rect',{'width':'100%','height':'100%','fill':'white'})
    pdf=fitz.open(); page=pdf.new_page(width=width,height=height)
    arial=Path('C:/Windows/Fonts/arialbd.ttf')
    page.insert_font(fontname='ArialBold',fontfile=str(arial))
    for stem,svg,x,y,w,h,label in layout:
        prefix=f'f{number}{label}_'
        idmap={el.attrib['id']:prefix+el.attrib['id'] for el in svg.iter() if 'id' in el.attrib}
        for el in svg.iter():
            for key,value in list(el.attrib.items()):
                if key=='id': el.set(key,idmap[value]); continue
                for old,new in idmap.items():
                    value=value.replace(f'url(#{old})',f'url(#{new})')
                    if value==f'#{old}': value=f'#{new}'
                el.set(key,value)
        svg.set('x',str(x));svg.set('y',str(y));svg.set('width',str(w));svg.set('height',str(h))
        root.append(svg)
        text=ET.SubElement(root,f'{{{NS}}}text',{'x':str(x+2),'y':str(y-6),
              'style':'font-family:Arial;font-weight:bold;font-size:18px'})
        text.text=label
        with fitz.open(folder/(stem+'.pdf')) as src:
            page.show_pdf_page(fitz.Rect(x,y,x+w,y+h),src,0)
        page.insert_text((x+2,y-6),label,fontname='ArialBold',fontsize=18)
    out=folder/f'figure_{number}_combined'
    for ext in ['.svg','.pdf','.png']: archive(out.with_suffix(ext))
    ET.ElementTree(root).write(out.with_suffix('.svg'),encoding='utf-8',xml_declaration=True)
    pdf.save(out.with_suffix('.pdf'))
    page.get_pixmap(dpi=180).save(out.with_suffix('.png'))
    page.get_pixmap(dpi=75).save(folder/f'figure_{number}_preview.png')
    pdf.close()
    assert not list(root.iter(f'{{{NS}}}image')), 'Raster image embedded in vector SVG'
    print('COMPOSED',out,flush=True)

def main():
    apply_arial_style()
    master=pd.read_csv(ROOT/'data/processed/archaea_swissprot_ec_master.csv')
    dataset=module('03_prepare_master_dataset.py')
    old=module('08_make_auc_figures.py')
    hom=module('11_summarize_homology_auc.py')
    interp=module('13_run_seven_class_interpretability.py')
    comp=module('14_mrmr_model_comparison.py')
    d1=RESULTS/'3.1_dataset_characterization'
    d2=RESULTS/'3.2_feature_selection_model_comparison'
    d3=RESULTS/'3.3_generalization_ladder'
    d5=RESULTS/'3.5_seven_class_interpretability'
    # Remove only the superseded active figure files, never experimental data.
    obsolete=['figure_4A_species_generalization_auc','figure_4B_species_per_class_auc',
              'figure_4C_species_roc_curves','figure_4D_homology_auc_ladder',
              'figure_4F_homology_roc_curves']
    for stem in obsolete:
        for ext in ['.png','.pdf','.svg']: archive(d3/(stem+ext),move=True)
    with export_all():
        dataset.figures(master,pd.read_csv(d1/'updated_dataset_summary.csv'),d1)
        table=pd.read_csv(d2/'algorithm_test_metrics.csv')
        ifs=pd.read_csv(d2/'ifs_validation_results.csv')
        predictions={}
        for name in comp.NAMES:
            frame=pd.read_csv(d2/'predictions'/f'{name.replace(" ","_")}_test.csv')
            predictions[name]=frame[[f'score_ec{i}' for i in range(1,8)]].to_numpy()
        truth=frame[[f'label_ec{i}' for i in range(1,8)]].to_numpy()
        # User-directed scientific model choice, not the validation winner.
        winner='LightGBM'
        comp.plots(table,ifs,truth,predictions,winner)
        old.generalization_figures(d3,old.load_outputs(RESULTS/'3.2_random_split_performance'),old.load_outputs(d3))
        old.plot_roc_curves(d3,pd.read_csv(d3/'optimized_lightgbm/optimized_test_probabilities.csv'),
            master,'figure_4B_species_roc_curves.png','ROC curves on the species-disjoint test set')
        hom.plot_ladder(pd.read_csv(d3/'homology_auc_summary.csv'),d3)
        experiments=[];class_rows=[]
        for t in range(90,0,-10):
            _,classes,pred=hom.read_experiment(d3/f'homology_id{t}')
            experiments.append((f'{t}% identity',pred))
            class_rows.append({'evaluation':f'Homology-{t}',**dict(zip(classes.class_name,classes.roc_auc))})
        matrix=pd.DataFrame(class_rows)
        archive(d3/'homology_per_class_auc.csv')
        matrix.to_csv(d3/'homology_per_class_auc.csv',index=False)
        hom.plot_roc_panels(experiments,master,d3)
        hom.plot_heatmap(matrix,d3)
        interp.plot_properties(pd.read_csv(d5/'class_physicochemical_effects.csv'),d5)
        interp.plot_pairwise_auc(pd.read_csv(d5/'pairwise_class_auc.csv'),d5)
        shap=pd.read_csv(d5/'class_specific_shap.csv'); selected=[]
        for ec in range(1,8):
            candidates=shap.loc[shap.ec_class.eq(ec)&shap.direction_stability.ge(.8)]
            if len(candidates)<3: candidates=shap.loc[shap.ec_class.eq(ec)]
            selected.extend(candidates.nlargest(3,'mean_absolute_shap').feature.tolist())
        interp.plot_shap(shap,list(dict.fromkeys(selected)),d5)
        interp.plot_network(pd.read_csv(d5/'multifunction_cooccurrence.csv'),
                            master.loc[master.fragment.fillna('').ne('fragment')],d5)
    compose(d1,2,[[('figure_2A_class_distribution',1),('figure_2B_length_distribution',1)],
                   [('figure_2C_taxonomic_distribution',1),('figure_2D_label_combinations',1)]])
    compose(d2,3,[[('figure_3A_algorithm_metrics',1),('figure_3B_algorithm_macro_roc',1)],
                   [('figure_3C_mrmr_ifs',1),('figure_3D_per_class_roc',1)]])
    # A/B at top; C, D and E vertically below, as explicitly requested.
    compose(d3,4,[[('figure_4A_species_per_class_auc',1),('figure_4B_species_roc_curves',1)],
                 [('figure_4C_homology_auc_ladder',1)],[('figure_4D_homology_roc_curves',1)],
                 [('figure_4E_homology_per_class_auc',1)]])
    compose(d5,6,[[('figure_6A_class_property_heatmap',1.5),('figure_6B_pairwise_auc_heatmap',1)],
                 [('figure_6C_class_shap_heatmap',1.5),('figure_6D_multifunction_network',1)]])
    (ROOT/'docs/figure_export_20260924.json').write_text(json.dumps({
        'panels':LOG,'font':'Arial','formats':['svg','pdf','png'],
        'combined_png_dpi':180,'individual_png_dpi':300,
        'homology_measured':[int(v.split('-')[1]) for v in pd.read_csv(d3/'homology_auc_summary.csv').dropna(subset=['macro_roc_auc']).evaluation if v.startswith('Homology-')],
        'homology_ticks':list(range(90,0,-10)),
        'font_scale':2,'panel_letter_font_scale':1,'main_model':'LightGBM (user-directed)',
        'font_audit':FONT_AUDIT,
        'missing_thresholds':'no current result; excluded 30% archive not reused; no fabricated results',
        'figure4_layout':'A/B top; C then D then E; historical LightGBM',
        'archive':str(ARCHIVE.relative_to(ROOT))},indent=2),encoding='utf-8')

if __name__=='__main__': main()
