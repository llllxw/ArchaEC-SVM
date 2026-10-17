"""Scoped, recoverable Figure 3C style revision; no training or score changes.

Run --render, visually inspect figure_3_preview.png, then --publish-reviewed.
"""
import argparse
import hashlib
import importlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'results/svm_migration_20261008'
SECTION = '3.2_feature_selection_model_comparison'
STAGE = RUN/'staged_results'/SECTION
FINAL = ROOT/'results'/SECTION
REV = RUN/'style_revisions/figure3c_svm_on_top_20261008'
NAMES = [f'{stem}.{ext}' for stem in ['figure_3C_mrmr_ifs','figure_3_combined']
         for ext in ['svg','pdf','png']] + ['figure_3_preview.png']
NOTE = ('Figure 3C: SVM line zorder=10, peak=11, linewidth=3.2; comparator '
        'lines=3, peaks=5, bands=1; bold SVM legend. All numerical values unchanged.')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def dump(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def backup(p):
    dest=REV/'before'/p.resolve().relative_to(ROOT.resolve())
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists(): shutil.copy2(p,dest)

def render():
    REV.mkdir(parents=True,exist_ok=False)
    manifest=read(RUN/'published_manifest.json')
    for row in manifest['files']:
        assert sha(ROOT/row['path'])==row['sha256'],row['path']
    snapshot={str(p.relative_to(ROOT)):sha(p) for p in STAGE.iterdir() if p.is_file()}
    snapshot.update({r['path']:r['sha256'] for r in manifest['files']})
    dump(REV/'before_hashes.json',snapshot)
    for folder in [STAGE,FINAL]:
        for name in NAMES+['five_fold_provenance.json','svm_figure3_provenance.json']:
            backup(folder/name)
    for ext in ['svg','pdf','png']: backup(ROOT/f'手稿/结果图/图3_SVM_五折.{ext}')
    for p in [RUN/'published_manifest.json',ROOT/'docs/final_svm_figures_manifest_20261008.json',
              ROOT/'docs/final_figures_manifest_20260925.json',RUN/'visual_qa.json',
              RUN/'report_validation.json',RUN/'execution_environment.json']:
        backup(p)
    mod=importlib.import_module('34_stage_svm_figure3')
    mod.main(ifs_only=True)
    changed={str((STAGE/name).relative_to(ROOT)) for name in NAMES}
    for path,h in snapshot.items():
        if path not in changed: assert sha(ROOT/path)==h,path
    dump(REV/'revision.json',{'status':'rendered; awaiting visual review','change':NOTE,
        'training_performed':False,'source_values_changed':False,
        'script_sha256':sha(ROOT/'scripts/34_stage_svm_figure3.py'),
        'figure6B':'unchanged; diagonal NaN/masked, not AUC zero',
        'rendered':{name:sha(STAGE/name) for name in NAMES}})
    print('Rendered; all published files and all non-C stage panels/data unchanged.',flush=True)

def publish():
    record=read(REV/'revision.json');assert record['status']=='rendered; awaiting visual review'
    snapshot=read(REV/'before_hashes.json')
    rendered={str((STAGE/name).relative_to(ROOT)) for name in NAMES}
    for path,h in snapshot.items():
        if path not in rendered: assert sha(ROOT/path)==h,path
    for name,h in record['rendered'].items(): assert sha(STAGE/name)==h,name
    for name in ['figure_3C_mrmr_ifs.svg','figure_3_combined.svg']:
        tree=ET.parse(STAGE/name);ns='{http://www.w3.org/2000/svg}'
        assert not list(tree.iter(ns+'image'))
        ids=[e.attrib['id'] for e in tree.iter() if 'id' in e.attrib]
        assert len(ids)==len(set(ids))
    for name in NAMES: shutil.copy2(STAGE/name,FINAL/name)
    for name in ['five_fold_provenance.json','svm_figure3_provenance.json']:
        doc=read(STAGE/name);doc['figure3C_style_revision']=NOTE
        dump(STAGE/name,doc);shutil.copy2(STAGE/name,FINAL/name)
    for ext in ['svg','pdf','png']:
        shutil.copy2(STAGE/f'figure_3_combined.{ext}',ROOT/f'手稿/结果图/图3_SVM_五折.{ext}')
    newhash=sha(STAGE/'figure_3_combined.svg')
    qa=read(RUN/'visual_qa.json');qa['svg_sha256']['3']=newhash
    qa['checks'].append(NOTE+' Revised C and full Figure 3 PNG visually reviewed.')
    dump(RUN/'visual_qa.json',qa)
    validation=read(RUN/'report_validation.json')
    for row in validation['figures']:
        if row['number']==3:row['svg_sha256']=newhash
    validation['post_validation_style_revision']={
        'note':NOTE,'all_published_numerical_tables_unchanged':True,
        'svg_vector_and_unique_ids_rechecked':True,'numerical_audit_rerun':False}
    dump(RUN/'report_validation.json',validation)
    env=read(RUN/'execution_environment.json')
    env['post_validation_plot_revision']={'note':NOTE,'script_sha256':record['script_sha256']}
    dump(RUN/'execution_environment.json',env)
    manifest=read(RUN/'published_manifest.json')
    allowed={str((FINAL/name).relative_to(ROOT)) for name in NAMES+
             ['five_fold_provenance.json','svm_figure3_provenance.json']}
    allowed.update(str((ROOT/f'手稿/结果图/图3_SVM_五折.{ext}').relative_to(ROOT)) for ext in ['svg','pdf','png'])
    changed=[]
    for row in manifest['files']:
        actual=sha(ROOT/row['path'])
        if actual!=row['sha256']:
            assert row['path'] in allowed,row['path']
            changed.append({'path':row['path'],'before':row['sha256'],'after':actual})
            row['sha256']=actual
    manifest['figure3C_style_revision']=NOTE
    dump(RUN/'published_manifest.json',manifest)
    dump(ROOT/'docs/final_svm_figures_manifest_20261008.json',manifest)
    index=read(ROOT/'docs/final_figures_manifest_20260925.json')
    for row in index['figures']:
        if row['figure']==3:row.update(sha256=newhash,style_revision=NOTE)
    dump(ROOT/'docs/final_figures_manifest_20260925.json',index)
    record.update(status='published after visual review',changed_published_files=changed,
                  published_files_verified=len(manifest['files']),backup=str((REV/'before').relative_to(ROOT)))
    dump(REV/'revision.json',record)
    print(f'Published Figure 3C style only; verified {len(manifest["files"])} manifest files.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--render',action='store_true');group.add_argument('--publish-reviewed',action='store_true')
    args=parser.parse_args()
    render() if args.render else publish()
