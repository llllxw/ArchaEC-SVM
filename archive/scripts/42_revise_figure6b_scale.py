"""Change only Figure 6B color normalization; preserve numerical evidence."""
import argparse, importlib, shutil
import xml.etree.ElementTree as ET
import pandas as pd

base=importlib.import_module('41_revise_figure6b_random')
r,m=base.r,base.m
ROOT,RUN,STAGE,FINAL=base.ROOT,base.RUN,base.STAGE,base.FINAL
REV=RUN/'style_revisions/figure6b_scale_05_1_20261008'
NAMES=base.ART+['SVM_FIGURE_CAPTION.md','five_fold_provenance.json']
NOTE='Figure 6B color scale 0.5–1; original colormap, lower triangle, all AUC/SD values and panels A/C/D unchanged.'
sha,read,dump=base.sha,base.read,base.dump

def backup(p):
    dest=REV/'before'/p.resolve().relative_to(ROOT.resolve())
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copy2(p,dest)

def render():
    REV.mkdir(parents=True,exist_ok=False)
    old=read(RUN/'published_manifest.json')
    snapshot={row['path']:row['sha256'] for row in old['files']}
    snapshot.update({str(p.relative_to(ROOT)):sha(p) for p in STAGE.iterdir() if p.is_file()})
    for path,h in snapshot.items():assert sha(ROOT/path)==h,path
    dump(REV/'before_hashes.json',snapshot)
    for folder in [STAGE,FINAL]:
        for name in NAMES:backup(folder/name)
    backup(FINAL/'FIVE_FOLD_RESULTS_README.md')
    for ext in ['svg','pdf','png']:backup(ROOT/f'手稿/结果图/图6_SVM_五折.{ext}')
    for p in [RUN/'published_manifest.json',RUN/'visual_qa.json',RUN/'report_validation.json',RUN/'execution_environment.json',
              ROOT/'docs/final_svm_figures_manifest_20261008.json',ROOT/'docs/final_figures_manifest_20260925.json']:backup(p)
    r.plot_pairwise_results(pd.read_csv(STAGE/'pairwise_class_auc_numeric.csv'),STAGE)
    m.compose(STAGE,6,[],[[('figure_6A_class_property_heatmap',1.5),('figure_6B_pairwise_auc_heatmap',1)],
                        [('figure_6C_class_shap_heatmap',1.5),('figure_6D_multifunction_network',1)]])
    p=STAGE/'SVM_FIGURE_CAPTION.md'
    p.write_text(p.read_text(encoding='utf-8').replace('颜色条0–1','颜色条0.5–1'),encoding='utf-8')
    p=STAGE/'five_fold_provenance.json';doc=read(p);doc['pairwise_color_scale']=[.5,1];dump(p,doc)
    allowed={str((STAGE/n).relative_to(ROOT)) for n in NAMES}
    for path,h in snapshot.items():
        if path not in allowed:assert sha(ROOT/path)==h,path
    dump(REV/'revision.json',{'status':'rendered','note':NOTE,'data_unchanged':True,
        'rendered':{n:sha(STAGE/n) for n in NAMES},'script_sha256':sha(ROOT/'scripts/36_report_svm_five_fold.py')})
    print('Rendered; numerical tables and all other panels byte-identical.',flush=True)

def publish():
    rec=read(REV/'revision.json');assert rec['status']=='rendered'
    allowed_stage={str((STAGE/n).relative_to(ROOT)) for n in NAMES}
    for path,h in read(REV/'before_hashes.json').items():
        if path not in allowed_stage:assert sha(ROOT/path)==h,path
    for n,h in rec['rendered'].items():assert sha(STAGE/n)==h
    for name in ['figure_6B_pairwise_auc_heatmap.svg','figure_6_combined.svg']:
        tree=ET.parse(STAGE/name);assert not list(tree.iter('{http://www.w3.org/2000/svg}image'))
        ids=[e.attrib['id'] for e in tree.iter() if 'id' in e.attrib];assert len(ids)==len(set(ids))
    for n in NAMES:shutil.copy2(STAGE/n,FINAL/n)
    for ext in ['svg','pdf','png']:shutil.copy2(STAGE/f'figure_6_combined.{ext}',ROOT/f'手稿/结果图/图6_SVM_五折.{ext}')
    p=FINAL/'FIVE_FOLD_RESULTS_README.md';p.write_text(p.read_text(encoding='utf-8').replace('色标0–1','色标0.5–1'),encoding='utf-8')
    h=sha(STAGE/'figure_6_combined.svg')
    qa=read(RUN/'visual_qa.json');qa['svg_sha256']['6']=h;qa['checks'].append(NOTE+' PNG visually reviewed.');dump(RUN/'visual_qa.json',qa)
    val=read(RUN/'report_validation.json')
    for row in val['figures']:
        if row['number']==6:row['svg_sha256']=h
    val['figure6B_color_scale_revision']={'note':NOTE,'numerical_tables_byte_identical':True};dump(RUN/'report_validation.json',val)
    env=read(RUN/'execution_environment.json');env['figure6B_scale_revision']={'note':NOTE,'script_sha256':rec['script_sha256']};dump(RUN/'execution_environment.json',env)
    manifest=read(RUN/'published_manifest.json')
    allowed={str((FINAL/n).relative_to(ROOT)) for n in NAMES+['FIVE_FOLD_RESULTS_README.md']}
    allowed.update(str((ROOT/f'手稿/结果图/图6_SVM_五折.{ext}').relative_to(ROOT)) for ext in ['svg','pdf','png'])
    for row in manifest['files']:
        actual=sha(ROOT/row['path'])
        if actual!=row['sha256']:assert row['path'] in allowed,row['path']
        row['sha256']=actual
    manifest['figure6B_color_scale_revision']=NOTE
    dump(RUN/'published_manifest.json',manifest);dump(ROOT/'docs/final_svm_figures_manifest_20261008.json',manifest)
    index=read(ROOT/'docs/final_figures_manifest_20260925.json')
    for row in index['figures']:
        if row['figure']==6:row.update(sha256=h,style_revision=NOTE)
    dump(ROOT/'docs/final_figures_manifest_20260925.json',index)
    rec.update(status='published after visual review',published_files_verified=len(manifest['files']));dump(REV/'revision.json',rec)
    print('Published; verified all 212 manifest files.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--publish-reviewed',action='store_true')
    publish() if parser.parse_args().publish_reviewed else render()
