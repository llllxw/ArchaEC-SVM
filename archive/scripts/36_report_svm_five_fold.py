"""Validated SVM report; keep old CV evidence and user templates immutable."""
from __future__ import annotations
import argparse, importlib, json, shutil, subprocess, sys, time
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

m=importlib.import_module('27_report_five_fold')
ROOT=m.ROOT; OLD=m.RUN; RUN=ROOT/'results/svm_migration_20261008'
m.RUN=RUN; m.STAGE=RUN/'staged_results'
original_load=m.load_task
def load(task):
    p,r=original_load(task.replace('lgb__','svm__',1))
    assert np.isclose(roc_auc_score(p[m.LABELS],p[m.SCORES],average='micro'),r['metrics']['micro_roc_auc'],rtol=0,atol=1e-12)
    if 'score_aux' in p:
        for key,val in [('auxiliary_roc_auc',roc_auc_score(p.label_aux,p.score_aux)),('auxiliary_ap',average_precision_score(p.label_aux,p.score_aux))]:
            assert np.isclose(val,r['metrics'][key],rtol=0,atol=1e-12)
    return p,r
m.load_task=load
m.original_section=lambda n: OLD/'staged_results'/m.SECTIONS[n]

def style():
    m.style();plt.rcParams.update({'font.size':22,'axes.labelsize':24,'axes.titlesize':25,
        'xtick.labelsize':20,'ytick.labelsize':20})

def dataset():
    folder=m.out(2);source=m.original_section(2)
    # Already validated descriptive tables: preserve exact source bytes.
    names={p.name for p in source.iterdir() if p.is_file()}
    for p in list(folder.glob('*.csv')):
        if p.name not in names:
            archive=RUN/'staging_previous_dataset';archive.mkdir(exist_ok=True)
            assert p.resolve().is_relative_to(RUN.resolve())
            target=archive/p.name
            if target.exists():raise FileExistsError(target)
            shutil.move(str(p),str(target))
    for p in source.iterdir():
        if p.is_file():shutil.copy2(p,folder/p.name)
    m.finish(2,{'descriptive_only':True,'source_panels_unchanged':True,'source_run':str(OLD.relative_to(ROOT)),
        'error_bars':'not applicable to whole-dataset counts'})

def comparison():
    # Separate interpreter isolates the old-report module's source-run globals.
    subprocess.run([sys.executable,str(ROOT/'scripts/34_stage_svm_figure3.py')],check=True,cwd=ROOT)
    folder=m.out(3)
    for p in (OLD/'staged_results'/m.SECTIONS[3]).glob('*.csv'):shutil.copy2(p,folder/p.name)
    provenance=json.loads((folder/'svm_figure3_provenance.json').read_text(encoding='utf-8'))
    provenance['status']='verified source predictions; staged for publication'
    m.dump(folder/'svm_figure3_provenance.json',provenance)
    m.finish(3,{'model':'RBF-SVM in D; all seven algorithms retained in A/B; six in C',
        'comparison_source':str(OLD.relative_to(ROOT)),'ifs':'inner validation only',
        'score_type':'sigmoid margins, not calibrated probabilities'})

def generalization():
    folder=m.out(4);frames={};rows=[];pc=[];aux=[];selection=[]
    for design in m.DESIGNS:
        loaded,oof=m.load_five(f'svm__{design}__');frames[design]=[p for p,r in loaded]
        oof.to_csv(folder/f'oof_{design}.csv',index=False)
        for f,(p,r) in enumerate(loaded):
            rows.append({'evaluation':design,'fold':f,**r['metrics']})
            selection.append({'evaluation':design,'fold':f,**r['selection']})
            aux.append({'evaluation':design,'fold':f,'roc_auc':r['metrics']['auxiliary_roc_auc'],'ap':r['metrics']['auxiliary_ap']})
            for c in range(7):pc.append({'evaluation':design,'fold':f,'ec_class':c+1,'class_name':m.CLASSES[c],
                'roc_auc':roc_auc_score(p[m.LABELS[c]],p[m.SCORES[c]]),'ap':average_precision_score(p[m.LABELS[c]],p[m.SCORES[c]])})
    s=m.summary(pd.DataFrame(rows),['evaluation'],m.METRICS+['micro_roc_auc'],folder,'homology_auc_summary').set_index('evaluation')
    pcs=m.summary(pd.DataFrame(pc),['evaluation','ec_class','class_name'],['roc_auc','ap'],folder,'homology_per_class_auc')
    m.summary(pd.DataFrame(aux),['evaluation'],['roc_auc','ap'],folder,'multifunction_auxiliary_metrics')
    pd.DataFrame(selection).to_csv(folder/'selected_features_by_fold.csv',index=False)
    style();fig,ax=plt.subplots(figsize=(15,8));x=np.arange(7)
    for design,delta,col,label in [('random',-.19,m.COLORS[1],'Random holdout'),('species',.19,m.COLORS[5],'Species-disjoint')]:
        sub=pcs[pcs.evaluation.eq(design)].sort_values('ec_class')
        ax.bar(x+delta,sub.roc_auc_mean,.34,yerr=sub.roc_auc_sd,color=col,alpha=.5,capsize=4,label=label)
        for i,r in enumerate(sub.itertuples()):ax.text(i+delta,r.roc_auc_mean+r.roc_auc_sd+.015,f'{r.roc_auc_mean:.3f}\n± {r.roc_auc_sd:.3f}',ha='center',fontsize=13)
    ax.set_xticks(x,[f'EC{i+1}\n{n}' for i,n in enumerate(m.CLASSES)],rotation=30,ha='right')
    ax.set(ylim=(0,1.2),ylabel='Class ROC-AUC (mean ± SD)',title='Class-level AUC across random and\nunseen-species tests')
    ax.legend(loc='upper center',ncol=2,fontsize=18,frameon=False)
    m.save(fig,folder,'figure_4A_species_per_class_auc')
    fig,ax=plt.subplots(figsize=(12,9))
    for c in range(7):m.roc(ax,frames['species'],f'EC{c+1} {m.CLASSES[c]}',m.COLORS[c],c)
    m.roc_axes(ax);ax.set_title('Species-disjoint: RBF-SVM');ax.legend(loc='lower right',fontsize=17)
    m.save(fig,folder,'figure_4B_species_roc_curves')
    thresholds=list(range(90,0,-10));designs=['random']+[f'homology_{t}' for t in thresholds]
    fig,ax=plt.subplots(figsize=(15,5));x=np.arange(10)
    for metric,col,label,offset,marker in [('ROC-AUC',m.COLORS[1],'Macro AUC',22,'o'),('micro_roc_auc',m.COLORS[5],'Micro AUC',-40,'s')]:
        mean=np.array([s.loc[d,metric+'_mean'] for d in designs]);sd=np.array([s.loc[d,metric+'_sd'] for d in designs])
        ax.errorbar(x,mean,yerr=sd,color=col,marker=marker,capsize=4,label=label)
        for xx,y,e in zip(x,mean,sd):ax.annotate(f'{y:.3f}\n± {e:.3f}',(xx,y),xytext=(0,offset),textcoords='offset points',ha='center',fontsize=14,color=col)
    ax.set_xticks(x,['Random']+[f'{t}%' for t in thresholds]);ax.set(ylim=(0,1.19),ylabel='ROC-AUC',xlabel='Random baseline / homology identity threshold',title='RBF-SVM across homology thresholds')
    ax.axhline(.8,color='#777777',ls=':',lw=1);ax.legend(loc='lower left',ncol=2,fontsize=18)
    m.save(fig,folder,'figure_4C_homology_auc_ladder')
    fig,axes=plt.subplots(3,3,figsize=(20,20))
    fig.suptitle('ROC curves under homology-cluster-disjoint evaluation',fontsize=25)
    for ax,t in zip(axes.flat,thresholds):
        for c in range(7):m.roc(ax,frames[f'homology_{t}'],f'EC{c+1}',m.COLORS[c],c)
        m.roc(ax,frames[f'homology_{t}'],'Macro','#222222',ls='--');m.roc_axes(ax)
        ax.set_title(f'{t}% identity');ax.legend(loc='lower right',fontsize=16)
    m.save(fig,folder,'figure_4D_homology_roc_curves')
    mat=[];ann=[]
    for t in thresholds:
        sub=pcs[pcs.evaluation.eq(f'homology_{t}')].sort_values('ec_class');mat.append(sub.roc_auc_mean)
        ann.append([f'{r.roc_auc_mean:.3f}\n± {r.roc_auc_sd:.3f}' for r in sub.itertuples()])
    fig,ax=plt.subplots(figsize=(15,11))
    sns.heatmap(np.array(mat),annot=np.array(ann),fmt='',cmap=m.REFERENCE_SEQUENTIAL_CMAP,vmin=0,vmax=1,
        linewidths=.8,xticklabels=[f'EC{i+1} {n}' for i,n in enumerate(m.CLASSES)],yticklabels=[f'{t}%' for t in thresholds],
        ax=ax,annot_kws={'size':15},cbar_kws={'label':'Mean ROC-AUC'})
    ax.set(xlabel='Enzyme class',ylabel='Homology identity threshold',title='Class-specific one-vs-rest ROC-AUC');ax.set_xticklabels(ax.get_xticklabels(),rotation=35,ha='right')
    m.save(fig,folder,'figure_4E_homology_per_class_auc')
    m.compose(folder,4,['figure_4A_species_per_class_auc','figure_4B_species_roc_curves','figure_4C_homology_auc_ladder','figure_4D_homology_roc_curves','figure_4E_homology_per_class_auc'])
    m.finish(4,{'model':'RBF-SVM; per-fold inner-training mRMR and validation IFS',
        'layout':'template: A/C/E left, B/D right','heatmap_scale':[0,1],
        'identity_10_20':'identical input cluster mappings, not independent difficulties'})

def supplementary(folder):
    source=m.module('12_run_multifunction_interpretability.py','svm_aux_report')
    data=pd.read_csv(ROOT/'data/processed/archaea_swissprot_ec_master.csv');split=pd.read_csv(RUN/'splits/random.csv')
    decode=[];thresholds=[];annotation=[];aux=[];examples=[]
    for f in range(5):
        p,rec=load(f'svm__random__{f}');task=RUN/'tasks'/f'svm__random__{f}'
        va=split[f'role_fold_{f}'].eq('validation').to_numpy();test=data.set_index('accession').loc[p.accession].reset_index()
        truth=p[m.LABELS].to_numpy();scores=p[m.SCORES].to_numpy();multi=p.label_aux.to_numpy();auxscore=p.score_aux.to_numpy()
        cut=np.array(rec['thresholds']);valaux=np.load(task/'validation_auxiliary.npy')
        grid=np.linspace(.01,.99,99);ym=data.loc[va,'is_multifunctional_top_level'].astype(int).to_numpy()
        acut=grid[np.argmax([f1_score(ym,valaux>=v,zero_division=0) for v in grid])]
        for c in range(7):thresholds.append({'fold':f,'ec_class':c+1,'validation_f1_threshold':cut[c]})
        fixed=source.force_nonempty(scores>=.5,scores);perclass=source.force_nonempty(scores>=cut,scores);guided=np.zeros_like(perclass)
        for i in range(len(p)):
            if auxscore[i]>=acut:
                guided[i]=perclass[i]
                if guided[i].sum()<2:guided[i,np.argsort(scores[i])[-2:]]=1
            else:guided[i,np.argmax(scores[i])]=1
        for name,pred in [('Fixed 0.50',fixed),('Validation class-specific',perclass),('Auxiliary-guided adaptive',guided)]:decode.append({'fold':f,**source.decoding_metrics(name,truth,pred)})
        aux.append({'fold':f,'model':'multifunction RBF-SVM','roc_auc':roc_auc_score(multi,auxscore),'ap':average_precision_score(multi,auxscore),'validation_threshold':acut})
        ann=source.annotation_auc_table(test,truth,scores);ann['fold']=f;ann['evaluable_class_mean_auc']=ann.macro_roc_auc
        ann.loc[ann.evaluable_classes.ne(7),'macro_roc_auc']=np.nan;annotation.extend(ann.to_dict('records'))
        exact=np.all(truth==guided,axis=1)
        # Descriptive score disagreement, not calibrated likelihood/confidence.
        disagreement=np.mean(np.abs(truth-scores),axis=1)
        for label,mask,reverse in [('Low-disagreement correct',exact,False),('High-disagreement incorrect',~exact,True),('Correct multifunctional',exact&(multi==1),False)]:
            eligible=np.flatnonzero(mask);order=eligible[np.argsort(disagreement[eligible])];order=order[::-1] if reverse else order
            for i in order[:5]:examples.append({'fold':f,'category':label,'accession':p.accession.iloc[i],
                'true_classes':';'.join(str(c+1) for c in np.flatnonzero(truth[i])),
                'predicted_classes':';'.join(str(c+1) for c in np.flatnonzero(guided[i])),
                'multifunction_uncalibrated_score':auxscore[i],'mean_absolute_score_disagreement':disagreement[i]})
    m.summary(pd.DataFrame(decode),['strategy'],['macro_f1','micro_f1','sample_f1','exact_match','hamming_loss'],folder,'decoding_comparison')
    m.summary(pd.DataFrame(thresholds),['ec_class'],['validation_f1_threshold'],folder,'adaptive_thresholds')
    m.summary(pd.DataFrame(aux),['model'],['roc_auc','ap','validation_threshold'],folder,'multifunction_auxiliary_metrics')
    m.summary(pd.DataFrame(annotation),['annotation','status'],['macro_roc_auc'],folder,'annotation_stratified_auc')
    pd.DataFrame(examples).to_csv(folder/'representative_proteins.csv',index=False,float_format='%.3f')
    pd.DataFrame(examples).to_csv(folder/'representative_proteins_full_precision.csv',index=False)

def ablation():
    for f in range(5):
        rec=json.loads((RUN/'tasks'/f'svm__random__{f}'/'metrics.json').read_text())
        assert rec['selection']['n_features']==432,'Full-representation baseline must actually use all 432 features'
    m.supplementary_multifunction=supplementary;m.ablation()
    m.finish(5,{'model':'RBF-SVM for all seven feature groups','paired_folds':True,
        'full_representation':'all 432 features selected in all five random folds',
        'feature_group_selection':'predefined complete feature groups; no IFS within ablations',
        'axis':'truncated lollipop axes; compare numerical values, not stem-length ratios'})

def pairwise_results(folder, design='random'):
    """Evaluate each unordered class pair separately in each outer test fold."""
    data=pd.read_csv(ROOT/'data/processed/archaea_swissprot_ec_master.csv');rows=[]
    loaded,_=m.load_five(f'svm__{design}__')
    for f,(p,_) in enumerate(loaded):
        frame=p.merge(data[['accession','n_top_classes','fragment']],on='accession',validate='one_to_one')
        frame=frame[frame.n_top_classes.eq(1)&frame.fragment.fillna('').ne('fragment')]
        for a,b in combinations(range(7),2):
            pair=frame[frame[m.LABELS[a]].eq(1)|frame[m.LABELS[b]].eq(1)];truth=pair[m.LABELS[a]].to_numpy()
            left=np.clip(pair[m.SCORES[a]],1e-7,1-1e-7);right=np.clip(pair[m.SCORES[b]],1e-7,1-1e-7)
            score=np.log(left/(1-left))-np.log(right/(1-right))
            val=roc_auc_score(truth,score) if len(np.unique(truth))==2 else np.nan
            rows.append({'class_a':a+1,'class_b':b+1,'fold':f,'pairwise_roc_auc':val,'n_a':int(truth.sum()),'n_b':int(len(truth)-truth.sum())})
    return m.summary(pd.DataFrame(rows),['class_a','class_b'],['pairwise_roc_auc'],folder,'pairwise_class_auc')

def plot_pairwise_results(pairwise,folder):
    style();matrix=np.full((7,7),np.nan);ann=np.full((7,7),'',dtype=object)
    for r in pairwise.itertuples():
        a,b=r.class_a-1,r.class_b-1;matrix[b,a]=r.pairwise_roc_auc_mean
        ann[b,a]=f'{r.pairwise_roc_auc_mean:.3f}\n± {r.pairwise_roc_auc_sd:.3f}' if np.isfinite(r.pairwise_roc_auc_mean) else 'NA'
    fig,ax=plt.subplots(figsize=(12,10))
    sns.heatmap(matrix,annot=ann,fmt='',mask=np.triu(np.ones((7,7),dtype=bool)),vmin=.5,vmax=1,
        cmap=m.REFERENCE_SEQUENTIAL_CMAP,linewidths=1,xticklabels=[f'EC{i}' for i in range(1,8)],yticklabels=[f'EC{i}' for i in range(1,8)],
        annot_kws={'size':16},cbar_kws={'label':'Pairwise ROC-AUC (mean)','ticks':np.linspace(.5,1,6)},ax=ax)
    ax.set(title='Random five-fold: RBF-SVM pairwise AUC',xlabel='Enzyme class',ylabel='Enzyme class')
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
    m.save(fig,folder,'figure_6B_pairwise_auc_heatmap')

def interpretability():
    folder=m.out(6);source=m.original_section(6)
    data=pd.read_csv(ROOT/'data/processed/archaea_swissprot_ec_master.csv')
    pairwise=pairwise_results(folder)
    # The separately documented representative-protein table retains its original population.
    _,oof=m.load_five('svm__homology_40__')
    rows=[];features=importlib.import_module('14_mrmr_model_comparison').FEATURES
    samplings=[]
    for f in range(5):
        d=RUN/'tasks'/f'shap__random__{f}';done=json.loads((d/'complete.json').read_text())
        assert all(m.sha(d/n)==h for n,h in done['outputs'].items())
        vals=np.load(d/'shap_summary.npy');samplings.append(json.loads((d/'sampling_manifest.json').read_text()))
        for c in range(7):
            for j,name in enumerate(features):rows.append({'fold':f,'ec_class':c+1,'class_name':m.CLASSES[c],'feature':name,
                'feature_group':'AAC' if j<20 else ('DPC' if j<420 else 'Physicochemical'),
                'mean_absolute_shap':vals[c,j,0],'positive_negative_contrast':vals[c,j,1]})
    m.dump(folder/'kernelshap_sampling_manifest.json',samplings)
    raw=pd.DataFrame(rows);shap=m.summary(raw,['ec_class','class_name','feature','feature_group'],['mean_absolute_shap','positive_negative_contrast'],folder,'class_specific_shap')
    grouped=raw.groupby(['fold','ec_class','class_name','feature_group'],as_index=False).mean_absolute_shap.sum()
    grouped['group_share']=grouped.mean_absolute_shap/grouped.groupby(['fold','ec_class']).mean_absolute_shap.transform('sum')
    m.summary(grouped,['ec_class','class_name','feature_group'],['mean_absolute_shap','group_share'],folder,'class_shap_group_summary')
    selected=list(dict.fromkeys(sum([shap[shap.ec_class.eq(c)].nlargest(3,'mean_absolute_shap_mean').feature.tolist() for c in range(1,8)],[])))
    normrows=[]
    for (f,c),g in raw[raw.feature.isin(selected)].groupby(['fold','ec_class']):
        scale=g.positive_negative_contrast.abs().max() or 1
        for r in g.itertuples():normrows.append({'fold':f,'ec_class':c,'feature':r.feature,'normalized_contrast':r.positive_negative_contrast/scale})
    norm=m.summary(pd.DataFrame(normrows),['ec_class','feature'],['normalized_contrast'],folder,'normalized_shap_contrast')
    examples=[];ef=oof.merge(data[['accession','protein_names','organism','fragment','n_top_classes']],on='accession',validate='one_to_one')
    ef=ef[ef.n_top_classes.eq(1)&ef.fragment.fillna('').ne('fragment')]
    for c in range(7):
        positive=ef[ef[m.LABELS[c]].eq(1)].sort_values(m.SCORES[c],ascending=False)
        for selection,subset in [('Highest-score annotated positive',positive.head(3)),('Lowest-score annotated positive',positive.tail(1))]:
            for rank,(_,r) in enumerate(subset.iterrows(),1):examples.append({'ec_class':c+1,'class_name':m.CLASSES[c],
                'selection':selection,'rank':rank,'fold':r.fold,'accession':r.accession,'protein_names':r.protein_names,'organism':r.organism,'uncalibrated_class_score':r[m.SCORES[c]]})
    pd.DataFrame(examples).to_csv(folder/'representative_class_proteins.csv',index=False,float_format='%.3f')
    pd.DataFrame(examples).to_csv(folder/'representative_class_proteins_full_precision.csv',index=False)
    for name in ['class_descriptive_statistics.csv','class_taxonomic_breadth.csv','class_physicochemical_effects.csv','multifunction_cooccurrence.csv']:
        for p in [source/name,source/(Path(name).stem+'_full_precision.csv')]:shutil.copy2(p,folder/p.name)
    for stem in ['figure_6A_class_property_heatmap','figure_6D_multifunction_network']:
        for ext in ['svg','pdf','png']:shutil.copy2(source/f'{stem}.{ext}',folder/f'{stem}.{ext}')
    plot_pairwise_results(pairwise,folder)
    mat=norm.pivot(index='ec_class',columns='feature',values='normalized_contrast_mean').loc[range(1,8),selected]
    sd=norm.pivot(index='ec_class',columns='feature',values='normalized_contrast_sd').loc[range(1,8),selected]
    annot=np.array([[f'{a:.3f}\n± {b:.3f}' for a,b in zip(r,s)] for r,s in zip(mat.values,sd.values)])
    fig,ax=plt.subplots(figsize=(max(18,len(selected)*1.05),10))
    sns.heatmap(mat,annot=annot,fmt='',annot_kws={'size':14},cmap=m.REFERENCE_DIVERGING_CMAP,center=0,vmin=-1,vmax=1,
        linewidths=.5,cbar_kws={'label':'Normalized KernelSHAP contrast (mean)'},ax=ax)
    ax.set(xlabel='Predictive feature',ylabel='EC class',title='Class-specific RBF-SVM KernelSHAP signatures\nRandom five-fold test samples (mean ± SD)')
    ax.set_yticklabels([f'EC{i}' for i in range(1,8)],rotation=0)
    ax.tick_params(axis='both',labelsize=24)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha='right');m.save(fig,folder,'figure_6C_class_shap_heatmap')
    m.compose(folder,6,[],[[('figure_6A_class_property_heatmap',1.5),('figure_6B_pairwise_auc_heatmap',1)],
        [('figure_6C_class_shap_heatmap',1.5),('figure_6D_multifunction_network',1)]])
    m.finish(6,{'model':'RBF-SVM','explanation':'KernelSHAP on raw decision margins, not TreeSHAP',
        'sampling':'per fold 14 training backgrounds and 42 complete single-label outer tests (6 per class)',
        'approximation':'1024 coalition samples; l1_reg=num_features(20); exploratory, no convergence claim',
        'pairwise_population':'complete single-label random five-fold outer tests','pairwise_score':'logit score difference',
        'pairwise_display':'lower triangle only; upper triangle and diagonal masked',
        'pairwise_color_scale':[.5,1],
        'descriptive_panels':['6A','6D'],'diagonal':'masked; not applicable, not zero or one'})

def audit():
    from importlib import metadata
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    import platform
    local=list(metadata.distributions(path=[str(ROOT/'.svm_dependencies')]))
    shap_dist=next(d for d in local if d.metadata['Name'].lower()=='shap')
    assert platform.python_version() in SpecifierSet(shap_dist.metadata['Requires-Python'])
    local_versions={d.metadata['Name'].lower():d.version for d in local};requirements=[]
    for raw in shap_dist.requires:
        r=Requirement(raw)
        if r.marker and not r.marker.evaluate({'extra':''}):continue
        version=local_versions.get(r.name.lower()) or metadata.version(r.name)
        assert version in r.specifier,(str(r),version,'incompatible declared SHAP dependency')
        requirements.append({'requirement':str(r),'installed':version,'compatible':True})
    checker=importlib.import_module('37_audit_svm_inputs');checker.main()
    trainer=importlib.import_module('35_svm_five_fold_experiments');count=0
    for task in trainer.tasks():
        folder=RUN/'tasks'/task;record=json.loads((folder/'complete.json').read_text())
        for n,h in record['outputs'].items():assert m.sha(folder/n)==h,(task,n);count+=1
    config=json.loads((RUN/'config.json').read_text(encoding='utf-8'))
    # prepare verifies original evidence and frozen config against current inputs.
    trainer.prepare()
    m.dump(RUN/'artifact_integrity_audit.json',{'status':'passed','tasks':90,'verified_file_hashes':count,'configuration_rechecked':True})
    packages={name:metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','matplotlib','joblib','threadpoolctl']}
    packages.update({d.metadata['Name']:d.version for d in metadata.distributions(path=[str(ROOT/'.svm_dependencies')])})
    m.dump(RUN/'execution_environment.json',{'python':sys.version,'platform':platform.platform(),'packages':packages,
        'shap_declared_dependencies_checked':requirements,
        'script_sha256':{p:m.sha(ROOT/'scripts'/p) for p in ['14_mrmr_model_comparison.py','34_stage_svm_figure3.py','35_svm_five_fold_experiments.py','36_report_svm_five_fold.py','37_audit_svm_inputs.py','38_watch_svm_reporting.py','39_prepare_compatible_shap.py']},
        'local_dependency_requirements_sha256':m.sha(ROOT/'requirements_svm_explanation.txt'),
        'recorded_at_validation':time.strftime('%Y-%m-%d %H:%M:%S')})
    checked=m.validate()
    for n in [4,5,6]:
        svg=(m.out(n)/f'figure_{n}_combined.svg').read_text(encoding='utf-8')
        assert 'LightGBM' not in svg and 'TreeSHAP' not in svg,(n,'obsolete model label')
    svg=(m.out(3)/'figure_3D_per_class_roc.svg').read_text(encoding='utf-8')
    assert 'RBF-SVM' in svg and 'LightGBM' not in svg
    checked['downstream_model_labels_verified']='SVM; no LightGBM or TreeSHAP in Figures 4-6; Figure 3D SVM'
    repair=RUN/'dependency_compatibility_repair.json'
    if repair.exists():
        record=json.loads(repair.read_text(encoding='utf-8'))
        for task,h in record['model_tasks_unchanged'].items():assert m.sha(RUN/'tasks'/task/'complete.json')==h
        differences=[]
        for f in range(5):
            task=RUN/'tasks'/f'shap__random__{f}'
            sampling=json.loads((task/'sampling_manifest.json').read_text())
            assert sampling['shap_version']==shap_dist.version
            old=ROOT/record['archive']/'tasks'/f'shap__random__{f}'/'shap_values.npy'
            differences.append(float(np.max(np.abs(np.load(old)-np.load(task/'shap_values.npy')))))
        record.update(status='verified: compatible explanations regenerated; all 85 model tasks unchanged',
            final_shap_version=shap_dist.version,per_fold_max_absolute_change_from_provisional_shap=differences)
        m.dump(repair,record);checked['dependency_compatibility_repair_verified']=True
    m.dump(RUN/'report_validation.json',checked)
    return checked

def report():
    general=pd.read_csv(m.out(4)/'homology_auc_summary.csv');abl=pd.read_csv(m.out(5)/'feature_ablation_auc_ap.csv')
    state=json.loads((RUN/'run_status.json').read_text(encoding='utf-8'))
    lines=['# 最终 RBF-SVM 五折结果更新','', '## Material Passport','',
        'Origin Skill: academic-research-suite / experiment-agent；Origin Mode: run + validate；Version: svm_results_v1；Verification: RECOMPUTED（由保存预测复算，非独立重新训练复现）。',
        '状态：实验完成并逐折复算。外层五折、内层分组留出验证；不是内外层均五折的嵌套交叉验证。此报告不代表独立外部验证。','',
        f'运行ID：svm_migration_20261008；训练状态：{state["status"]}；累计训练耗时：{state["elapsed_seconds"]:.1f}秒；上限21600秒；90/90任务。命令：E:/anaconda3/python.exe -u scripts/35_svm_five_fold_experiments.py --hours 6；工作目录：{ROOT}。',
        '总体解释置信等级：CAUTION（开发集复用、折间训练集重叠及近似抽样解释）；算术与输入核查通过不意味着独立验证或因果确认。统计误区扫描11/11，细节见statistical_validation_notes.md。','',
        '## 模型和评估口径','',
        '20维AAC + 400维DPC + 12维理化特征 → 内部训练集mRMR排序 → 内部验证IFS选择16/32/64/128/256/432维 → 训练集StandardScaler → 七个独立类别平衡RBF-SVC（C=2，gamma=scale）。外层测试样本不进入对应折的拟合和IFS选择。随机五折均选择432维；物种/同源折的特征数见 selected_features_by_fold.csv。',
        '实际IFS过程对每个候选前缀分别在内部训练集拟合标准化和SVM，再比较内部验证AUC，随后冻结所选前缀模型用于外层测试。随机五折均保留全部432维，因此本结果支持特征排序/数量评估，不支持声称随机条件下已经实现维数压缩。',
        'sigmoid(decision_function)只是0–1评分，不是校准概率，也不要求七类之和为1。辅助二分类SVM独立识别多功能标签，不是共享参数的多任务神经网络；辅助引导解码是另行报告的策略，不冒充主任务AUC的来源。五折模型用于评估，不自动构成部署集成。',
        '与原五折实验保持同一数据、折、种子；同序列不得跨折；物种/同源组隔离。每折约64%用于拟合、16%内部验证、20%外层测试，数量受分组约束影响。没有使用外层测试集选特征、阈值或超参数。',
        '所有性能为五个外层测试折的算术均值 ± 样本标准差（ddof=1），三位小数。AUC/AP先对七类宏平均；Accuracy为多标签完全匹配准确率。ROC阴影是TPR折间SD，图例是原始折AUC均值±SD；二者不是同一统计对象。SD不是置信区间，训练集重叠，不据微小差值宣称统计显著。',
        'Micro AUC跨类别合并评分计算；由于不同OvR输出未校准，其跨类别比较需谨慎。成对AUC由既有七分类器在对应两类样本上的决策分数差得到，并非重新训练21个二分类模型，也不是校准后的成对概率。',
        '原始10%和20%同源簇映射完全相同，折划分也相同，不是两种独立难度。已有多轮数据开发，本次交叉验证不能消除历史分析选择偏倚。','',
        '最终SVM是在已有算法比较之后由用户指定，未新增独立的最终确认测试集；因此这些五折值属于开发阶段的比较与泛化评估，不应宣称是完全未参与算法选择的无偏最终确认结果。','',
        '## 泛化结果','', '| 条件 | Macro AUC | Macro AP |','|---|---:|---:|']
    for _,r in general.iterrows():lines.append(f'| {r["evaluation"]} | {r["ROC-AUC"]} | {r["AP"]} |')
    lines+=['','## 特征消融','', '| 特征组 | Macro AUC | Macro AP |','|---|---:|---:|']
    for _,r in abl.iterrows():lines.append(f'| {r["feature_set"]} | {r["ROC-AUC"]} | {r["AP"]} |')
    lines+=['','## 图表与解释范围','',
        '图3A/B保留七算法比较，图3C是六种表征模型的mRMR-IFS（原始序列CNN不适用），图3D为SVM七类ROC。图4按模板左A/C/E、右B/D排列；包含随机、物种及90%至10%同源五折。图5七种特征组合统一SVM，使用完整预定义组，不在消融中再次运行IFS。图6A/D为原始数据描述；B为随机五折外层测试完整单标签样本的成对AUC，仅展示21组下三角，上三角及对角线遮罩；C为SVM KernelSHAP，不沿用TreeSHAP。',
        'KernelSHAP每折预先按真实类别分层抽取14个训练背景样本（每类2个）和42个完整单标签外层测试样本（每类6个），合计210个解释样本。解释原始决策间隔，1024次coalition采样、l1_reg=num_features(20)。保存样本ID、种子、数值等价和加性重建核查。加性核查不证明采样收敛；背景小、局部稀疏近似、单标签限制均影响解释，不能视为总体精确贡献或催化机制。参见[SHAP官方KernelExplainer说明](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html)。',
        '图6C特征为各类别平均绝对KernelSHAP前三名的并集；对比是正类与其余类解释均值之差，逐折按本类所示特征最大绝对差归一化，再汇总均值和SD。特征选择用于事后解释展示，不作为泛化性能证据。',
        '解释样本为类别平衡抽样，不代表原数据的自然类别比例。AAC、DPC和理化量存在相关及组成约束，背景替换产生的混合特征未必对应真实蛋白；归因应理解为该解释设置下的模型行为。各特征组维数不同，组SHAP总量不宜直接解释为生物学重要性的公平排名。',
        '归因结果的SD反映五折模型、背景和解释样本的综合变化，不是KernelSHAP蒙特卡罗采样标准误，也不是置信区间。',
        '依赖复核曾发现SHAP 0.52.0的NumPy>=2要求与现有NumPy 1.26.4不符。初次解释仅作为中间结果归档，随后使用声明依赖全部核验通过的SHAP 0.49.1重新计算五折解释；数据、抽样、种子、1024采样及稀疏设置均未改变，85项模型训练/消融任务校验哈希保持不变，累计运行上限未重置。详见dependency_compatibility_repair.json。',
        'Arial、原配色和用户模板布局保留，性能表显示三位小数±SD，*_folds.csv与*_numeric.csv保留完整精度。图2与图6A/D没有人工添加CV标准差。原模板PNG/AI未覆盖，新输出带SVM标识。',
        '训练及逐类/逐前缀/逐解释样本检查点位于本运行tasks目录；旧LightGBM运行保留，正式目录替换前完整归档至 results/_archive_pre_svm_20261008。旧手稿与旧总览图中的LightGBM、TreeSHAP或校准概率说法不能直接用于当前结果；本轮未重写整篇手稿。']
    (RUN/'experiment_result.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    captions={
      2:'图2 数据集组成。A：七类EC标签计数，多标签蛋白可计入多个类别。B：序列长度分布，黑线为既有中位数标记。C：分类学组成。D：标签组合。均为全体数据的描述统计，不附加人为的交叉验证SD。',
      3:'图3 特征筛选与算法比较。A：七算法八项外层测试指标，柱高为五折均值，误差线为样本SD，透明度0.5；Accuracy指多标签exact match。B：七算法宏平均ROC。C：内部验证mRMR-IFS，六种表征模型参与，原始序列1D-CNN不适用；星号为平均验证AUC最大值，图例显示[特征数，AUC±SD]，不用于外层测试选参。D：最终RBF-SVM的七类ROC。ROC阴影为共同FPR网格上的TPR折间SD，图例为原始折AUC均值±SD。',
      4:'图4 RBF-SVM泛化性能。A：随机与物种隔离逐类AUC双柱，透明度0.5，误差线为五折SD。B：物种隔离逐类ROC。C：Random基线及90%至10%同源隔离的宏/微AUC，所有点标注均值±SD。D：九个同源阈值逐类及宏平均ROC，阴影为TPR折间SD。E：逐类同源AUC热图（0–1），显示均值±SD。左列A/C/E、右列B/D。10%与20%的原始簇映射一致，不是独立难度；同源阈值是聚类设置，不等同于任意跨折序列对一致性的严格上界。',
      5:'图5 RBF-SVM特征组消融。A：宏ROC-AUC。B：宏AP，AP由average_precision_score计算，不是梯形积分PR曲线面积。七种预定义特征组合在同一随机五折与相同模型设置下评估；点和误差线为均值±SD。完整表征432维；AAC20维、DPC400维、理化12维及两两组合。全表征在五个随机折IFS均选432维，其他消融使用完整预定义组，不再执行IFS。灰/冷/暖色分别代表全组/单组/双组，透明度0.5。纵轴截断，棒长度不能作为效果比值。',
      6:'图6 七类酶差异与RBF-SVM预测解释。A：全数据理化属性描述性差异。B：随机划分五折外层测试完整单标签蛋白的21组成对AUC，仅显示下三角，上三角及对角线留白（非0或1），颜色条0.5–1。各折分别使用对应两类样本及既有SVM评分的logit差计算AUC，再汇总五折均值±样本SD（ddof=1），保留三位小数；不是合并五折预测后计算单个AUC，也不是重新训练21个分类器。C：随机五折抽样外层测试KernelSHAP对比，解释原始SVM间隔；每折14个训练背景、42个解释样本，每类分别2个与6个。1024次采样、局部稀疏约束20特征。展示各类绝对贡献前三特征的并集，正类/其余类贡献均值差逐折归一化后计算均值±SD。D：真实EC标签共现，非模型预测；A/D不人为添加CV标准差。SHAP与共现仅支持预测/描述关联，不证明因果或催化机制。'
    }
    for number,caption in captions.items():
        (m.out(number)/'SVM_FIGURE_CAPTION.md').write_text(caption+'\n',encoding='utf-8')

def publish():
    qa=json.loads((RUN/'visual_qa.json').read_text(encoding='utf-8'));assert qa['status']=='passed'
    for n in range(2,7):assert qa['svg_sha256'][str(n)]==m.sha(m.out(n)/f'figure_{n}_combined.svg')
    audit();report();backup=ROOT/'results/_archive_pre_svm_20261008';manifest=[];retired=[]
    for number in range(2,7):
        src=m.out(number);dest=ROOT/'results'/m.SECTIONS[number];archive=backup/m.SECTIONS[number]
        for name in ['five_fold_provenance.json','svm_figure3_provenance.json']:
            p=src/name
            if p.exists():
                provenance=json.loads(p.read_text(encoding='utf-8'));provenance['status']='published after numerical validation and visual QA'
                m.dump(p,provenance)
        archive.mkdir(parents=True,exist_ok=True);incoming={p.name for p in src.iterdir() if p.is_file()}
        for p in dest.iterdir():
            if p.is_file() and not (archive/p.name).exists():shutil.copy2(p,archive/p.name)
        for p in list(dest.iterdir()):
            if p.is_file() and p.name not in incoming and p.name!='FIVE_FOLD_RESULTS_README.md':
                legacy=dest/'legacy_lightgbm_cv';legacy.mkdir(exist_ok=True);target=legacy/p.name
                assert p.resolve().is_relative_to(ROOT.resolve()) and target.resolve().is_relative_to(ROOT.resolve())
                if target.exists():raise FileExistsError(target)
                assert m.sha(p)==m.sha(archive/p.name);shutil.move(str(p),str(target));retired.append(str(target.relative_to(ROOT)))
        for p in src.iterdir():
            if p.is_file():shutil.copy2(p,dest/p.name);manifest.append({'path':str((dest/p.name).relative_to(ROOT)),'sha256':m.sha(dest/p.name)})
        (dest/'FIVE_FOLD_RESULTS_README.md').write_text('当前正式结果：results/svm_migration_20261008。最终模型RBF-SVM；图3保留七算法比较。外层五折、内部验证选特征，均值±样本SD(ddof=1)，三位小数；描述性面板不添加SD。SVM输出为未校准评分；图6C为抽样KernelSHAP，不是TreeSHAP。重绘使用scripts/36_report_svm_five_fold.py，训练使用scripts/35_svm_five_fold_experiments.py。旧LightGBM证据保留于results/five_fold_cv_20261007和results/_archive_pre_svm_20261008；未更新旧顶层文件归入legacy_lightgbm_cv，仅作历史追溯。详见新运行experiment_result.md。\n',encoding='utf-8')
        manifest.append({'path':str((dest/'FIVE_FOLD_RESULTS_README.md').relative_to(ROOT)),
            'sha256':m.sha(dest/'FIVE_FOLD_RESULTS_README.md')})
        if number>=3:
            for ext in ['svg','pdf','png']:
                target=ROOT/'手稿/结果图'/f'图{number}_SVM_五折.{ext}';shutil.copy2(src/f'figure_{number}_combined.{ext}',target)
                manifest.append({'path':str(target.relative_to(ROOT)),'sha256':m.sha(target)})
    record={'status':'published; SVM numerically verified, template-aligned; awaiting user visual approval',
        'source_run':str(RUN.relative_to(ROOT)),'files':manifest,'backup':str(backup.relative_to(ROOT)),'retired':retired}
    m.dump(RUN/'published_manifest.json',record);m.dump(ROOT/'docs/final_svm_figures_manifest_20261008.json',record)
    previous=ROOT/'docs/final_figures_manifest_20260925.json'
    if previous.exists():
        saved=backup/'docs'/previous.name;saved.parent.mkdir(parents=True,exist_ok=True)
        if not saved.exists():shutil.copy2(previous,saved)
        current=json.loads(previous.read_text(encoding='utf-8'))
        current.update(date=time.strftime('%Y-%m-%d'),status=record['status'],source_run=str(RUN.relative_to(ROOT)),
            final_model='RBF-SVM',explanation='sampled KernelSHAP of raw decision margins',
            score_type='uncalibrated sigmoid decision scores',previous_version=str(saved.relative_to(ROOT)))
        for item in current['figures']:
            item['sha256']=m.sha(ROOT/item['path']);item['style_revision']='SVM five-fold; user templates 3/4/6; Arial; mean ± sample SD'
        m.dump(previous,current)
    m.dump(RUN/'completion_status.json',{'status':'complete_and_published','time':time.strftime('%Y-%m-%d %H:%M:%S')})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['dataset','comparison','generalization','ablation','interpretability','all','validate','publish'],default='all');parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    functions={'dataset':dataset,'comparison':comparison,'generalization':generalization,'ablation':ablation,'interpretability':interpretability}
    for name,fn in functions.items():
        if args.stage in ['all',name]:fn();print('STAGED',name,flush=True)
    if args.stage in ['all','validate']:audit();report()
    if args.publish or args.stage=='publish':publish()
