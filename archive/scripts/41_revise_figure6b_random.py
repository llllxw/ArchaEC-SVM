"""Recompute Figure 6B from existing random outer-test predictions; no training.

Run --render, inspect the PNGs, then --publish-reviewed. Old Homology-40
tables and figures are recoverably copied into this revision's before/ folder.
"""
import argparse
import importlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

r=importlib.import_module('36_report_svm_five_fold')
m=r.m;ROOT=r.ROOT;RUN=r.RUN;STAGE=m.out(6);FINAL=ROOT/'results'/m.SECTIONS[6]
REV=RUN/'style_revisions/figure6b_random_lower_20261008'
ART=[f'{stem}.{ext}' for stem in ['figure_6B_pairwise_auc_heatmap','figure_6_combined']
     for ext in ['svg','pdf','png']]+['figure_6_preview.png']
TABLES=['pairwise_class_auc.csv','pairwise_class_auc_folds.csv','pairwise_class_auc_numeric.csv']
NAMES=ART+TABLES+['SVM_FIGURE_CAPTION.md','five_fold_provenance.json']
NOTE='Figure 6B: random five-fold outer-test pairwise AUC; lower triangle only; mean ± sample SD, 3 decimals; no retraining.'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=m.sha
dump=m.dump

def backup(p):
    dest=REV/'before'/p.resolve().relative_to(ROOT.resolve())
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copy2(p,dest)

def validate_values():
    """Independent rank-sum AUC check for all 105 fold/pair results."""
    raw=pd.read_csv(STAGE/'pairwise_class_auc_folds.csv',float_precision='round_trip')
    numeric=pd.read_csv(STAGE/'pairwise_class_auc_numeric.csv',float_precision='round_trip')
    display=pd.read_csv(STAGE/'pairwise_class_auc.csv')
    master=pd.read_csv(ROOT/'data/processed/archaea_swissprot_ec_master.csv').set_index('accession')
    sources={};fold_sizes=[];checks=0
    for f in range(5):
        path=RUN/'tasks'/f'svm__random__{f}'/'predictions.csv'
        sources[str(path.relative_to(ROOT))]=sha(path)
        p=pd.read_csv(path,float_precision='round_trip').set_index('accession')
        meta=master.loc[p.index]
        p=p.loc[meta.n_top_classes.eq(1)&meta.fragment.fillna('').ne('fragment')]
        fold_sizes.append({'fold':f,'eligible_proteins':len(p)})
        for row in raw[raw.fold.eq(f)].itertuples():
            a,b=int(row.class_a)-1,int(row.class_b)-1
            pair=p.loc[p[m.LABELS[a]].eq(1)|p[m.LABELS[b]].eq(1)]
            pos=pair[m.LABELS[a]].to_numpy(bool);n1=int(pos.sum());n0=len(pos)-n1
            assert n1==row.n_a and n0==row.n_b and n1>0 and n0>0
            left=np.clip(pair[m.SCORES[a]].to_numpy(),1e-7,1-1e-7)
            right=np.clip(pair[m.SCORES[b]].to_numpy(),1e-7,1-1e-7)
            score=np.log(left/(1-left))-np.log(right/(1-right))
            value=(rankdata(score,method='average')[pos].sum()-n1*(n1+1)/2)/(n1*n0)
            assert abs(value-row.pairwise_roc_auc)<1e-12
            checks+=1
    assert checks==105 and len(numeric)==len(display)==21
    for row in numeric.itertuples():
        values=raw.loc[raw.class_a.eq(row.class_a)&raw.class_b.eq(row.class_b),'pairwise_roc_auc'].to_numpy()
        assert len(values)==5 and np.isfinite(values).all() and row.pairwise_roc_auc_n_folds==5
        mean,sd=values.mean(),values.std(ddof=1)
        assert abs(mean-row.pairwise_roc_auc_mean)<1e-12 and abs(sd-row.pairwise_roc_auc_sd)<1e-12
        shown=display.loc[display.class_a.eq(row.class_a)&display.class_b.eq(row.class_b),'pairwise_roc_auc'].item()
        assert shown==f'{mean:.3f} ± {sd:.3f}'
    return {'fold_pair_auc_rank_sum_checks':checks,'summary_pairs_verified':21,
            'source_prediction_sha256':sources,'eligible_proteins_by_fold':fold_sizes,
            'all_pairs_estimable_in_all_folds':True,'sd_ddof':1}

def render():
    REV.mkdir(parents=True,exist_ok=False)
    manifest=read(RUN/'published_manifest.json')
    snapshot={row['path']:row['sha256'] for row in manifest['files']}
    snapshot.update({str(p.relative_to(ROOT)):sha(p) for p in STAGE.iterdir() if p.is_file()})
    for path,h in snapshot.items():assert sha(ROOT/path)==h,path
    dump(REV/'before_hashes.json',snapshot)
    for folder in [STAGE,FINAL]:
        for name in NAMES:backup(folder/name)
    backup(FINAL/'FIVE_FOLD_RESULTS_README.md')
    for ext in ['svg','pdf','png']:backup(ROOT/f'手稿/结果图/图6_SVM_五折.{ext}')
    for p in [RUN/'published_manifest.json',RUN/'visual_qa.json',RUN/'report_validation.json',
              RUN/'execution_environment.json',RUN/'experiment_result.md',
              ROOT/'docs/final_svm_figures_manifest_20261008.json',ROOT/'docs/final_figures_manifest_20260925.json']:
        backup(p)
    pairwise=r.pairwise_results(STAGE,design='random')
    validation=validate_values()
    r.plot_pairwise_results(pairwise,STAGE)
    m.compose(STAGE,6,[],[[('figure_6A_class_property_heatmap',1.5),('figure_6B_pairwise_auc_heatmap',1)],
        [('figure_6C_class_shap_heatmap',1.5),('figure_6D_multifunction_network',1)]])
    r.report()
    provenance=read(STAGE/'five_fold_provenance.json')
    provenance.update(pairwise_population='complete single-label random five-fold outer tests',
        pairwise_display='lower triangle only; upper triangle and diagonal masked',
        diagonal='not applicable; masked, not zero or one',pairwise_revision=NOTE,
        pairwise_previous_version=str((REV/'before').relative_to(ROOT)))
    dump(STAGE/'five_fold_provenance.json',provenance)
    allowed={str((STAGE/name).relative_to(ROOT)) for name in NAMES}
    for path,h in snapshot.items():
        if path not in allowed:assert sha(ROOT/path)==h,path
    dump(REV/'revision.json',{'status':'rendered; awaiting visual review','change':NOTE,
        'validation':validation,'new_training':False,'other_panels_unchanged':True,
        'script_sha256':sha(ROOT/'scripts/36_report_svm_five_fold.py'),
        'rendered':{name:sha(STAGE/name) for name in NAMES}})
    print(pairwise[['class_a','class_b','pairwise_roc_auc_mean','pairwise_roc_auc_sd']].to_string(index=False),flush=True)
    print('Validated 105 fold/pair AUCs by rank sums; rendered C-independent Figure 6 update.',flush=True)

def publish():
    rec=read(REV/'revision.json');assert rec['status']=='rendered; awaiting visual review'
    snapshot=read(REV/'before_hashes.json')
    allowed_stage={str((STAGE/name).relative_to(ROOT)) for name in NAMES}
    for path,h in snapshot.items():
        if path not in allowed_stage:assert sha(ROOT/path)==h,path
    for name,h in rec['rendered'].items():assert sha(STAGE/name)==h,name
    for name in ['figure_6B_pairwise_auc_heatmap.svg','figure_6_combined.svg']:
        tree=ET.parse(STAGE/name);assert not list(tree.iter('{http://www.w3.org/2000/svg}image'))
        ids=[e.attrib['id'] for e in tree.iter() if 'id' in e.attrib];assert len(ids)==len(set(ids))
    for name in NAMES:shutil.copy2(STAGE/name,FINAL/name)
    for ext in ['svg','pdf','png']:
        shutil.copy2(STAGE/f'figure_6_combined.{ext}',ROOT/f'手稿/结果图/图6_SVM_五折.{ext}')
    readme=FINAL/'FIVE_FOLD_RESULTS_README.md'
    readme.write_text(readme.read_text(encoding='utf-8')+'\n图6B已更新为随机划分五折外层测试的21组成对AUC下三角热图。上三角与对角线留白，五折均值±样本SD（ddof=1），三位小数，色标0–1。旧Homology-40图表保留于'+str((REV/'before').relative_to(ROOT))+'。\n',encoding='utf-8')
    newhash=sha(STAGE/'figure_6_combined.svg')
    qa=read(RUN/'visual_qa.json');qa['svg_sha256']['6']=newhash
    qa['checks']=[x for x in qa['checks'] if not x.startswith('Figure 6 ')]
    qa['checks'].append(NOTE+' Panel B and combined PNG visually reviewed; A/C/D unchanged.')
    dump(RUN/'visual_qa.json',qa)
    val=read(RUN/'report_validation.json')
    for row in val['figures']:
        if row['number']==6:row['svg_sha256']=newhash
    val['figure6B_revision_validation']=rec['validation']
    dump(RUN/'report_validation.json',val)
    env=read(RUN/'execution_environment.json')
    env['figure6B_revision']={'note':NOTE,'script_sha256':rec['script_sha256']}
    dump(RUN/'execution_environment.json',env)
    manifest=read(RUN/'published_manifest.json');changed=[]
    allowed={str((FINAL/name).relative_to(ROOT)) for name in NAMES+['FIVE_FOLD_RESULTS_README.md']}
    allowed.update(str((ROOT/f'手稿/结果图/图6_SVM_五折.{ext}').relative_to(ROOT)) for ext in ['svg','pdf','png'])
    for row in manifest['files']:
        actual=sha(ROOT/row['path'])
        if actual!=row['sha256']:
            assert row['path'] in allowed,row['path']
            changed.append({'path':row['path'],'before':row['sha256'],'after':actual});row['sha256']=actual
    manifest['figure6B_revision']=NOTE
    dump(RUN/'published_manifest.json',manifest);dump(ROOT/'docs/final_svm_figures_manifest_20261008.json',manifest)
    index=read(ROOT/'docs/final_figures_manifest_20260925.json')
    for row in index['figures']:
        if row['figure']==6:row.update(sha256=newhash,style_revision=NOTE)
    dump(ROOT/'docs/final_figures_manifest_20260925.json',index)
    rec.update(status='published after numerical verification and visual review',
               changed_published_files=changed,published_files_verified=len(manifest['files']))
    dump(REV/'revision.json',rec)
    print(f'Published; verified {len(manifest["files"])} manifest files; original templates unchanged.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--render',action='store_true');group.add_argument('--publish-reviewed',action='store_true')
    args=parser.parse_args();render() if args.render else publish()
