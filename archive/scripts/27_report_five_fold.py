"""Recompute fold metrics, write mean +/- sample SD, and stage figures 2--6.

No model training or performance-based split selection. Publish is explicit and
requires all sections plus validation; historical artifacts are recoverable.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, shutil, sys, time
from itertools import combinations
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, f1_score
from plot_style import apply_arial_style, REFERENCE_PALETTE, REFERENCE_SEQUENTIAL_CMAP, REFERENCE_DIVERGING_CMAP

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/five_fold_cv_20261007'
STAGE=RUN/'staged_results'
SECTIONS={2:'3.1_dataset_characterization',3:'3.2_feature_selection_model_comparison',
          4:'3.3_generalization_ladder',5:'3.4_multifunction_interpretability',6:'3.5_seven_class_interpretability'}
LABELS=[f'label_ec{i}' for i in range(1,8)];SCORES=[f'score_ec{i}' for i in range(1,8)]
CLASSES=['Oxidoreductases','Transferases','Hydrolases','Lyases','Isomerases','Ligases','Translocases']
NAMES=['Logistic Regression','RBF-SVM','Random Forest','XGBoost','LightGBM','DNN','1D-CNN']
METRICS=['ROC-AUC','AP','Accuracy','Precision','Recall','F1','Specificity','MCC']
DESIGNS=['random','species']+[f'homology_{v}' for v in range(10,100,10)]
GRID=np.linspace(0,1,501)
COLORS=REFERENCE_PALETTE

def module(file,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@lru_cache(maxsize=1)
def master_truth():
    return pd.read_csv(ROOT/'data/processed/archaea_swissprot_ec_master.csv',usecols=['accession']+LABELS).set_index('accession')
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def out(number):p=STAGE/SECTIONS[number];p.mkdir(parents=True,exist_ok=True);return p
def original_section(number):
    archived=ROOT/'results/_archive_pre_five_fold_20261007'/SECTIONS[number]
    return archived if (RUN/'published_manifest.json').exists() and archived.exists() else ROOT/'results'/SECTIONS[number]
def style():
    apply_arial_style();plt.rcParams.update({'font.size':18,'axes.labelsize':20,'axes.titlesize':20,
       'xtick.labelsize':16,'ytick.labelsize':16,'legend.fontsize':15,'svg.fonttype':'none','pdf.fonttype':42})
def save(fig,folder,stem):
    fig.tight_layout(pad=1.3)
    for axis in fig.axes:
        for artist in axis.collections:artist.set_rasterized(False)
    for ext in ['svg','pdf','png']:fig.savefig(folder/f'{stem}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
def compose(folder,number,stems,rows=None):
    exp=module('17_export_composite_figures.py','cv_composer');exp.ARCHIVE=RUN/'staging_previous'
    rows=rows or [[(s,1) for s in stems[i:i+2]] for i in range(0,len(stems),2)]
    exp.compose(folder,number,rows)
def summary(raw,groups,metrics,folder,stem):
    raw.to_csv(folder/f'{stem}_folds.csv',index=False)
    rows=[];show=[]
    for key,g in raw.groupby(groups,sort=False,dropna=False):
        key=key if isinstance(key,tuple) else (key,)
        row=dict(zip(groups,key));display=row.copy()
        assert len(g)==5 and sorted(g.fold.tolist())==list(range(5)),(stem,key,'not five distinct folds')
        for metric in metrics:
            v=g[metric].to_numpy(float);n=int(np.isfinite(v).sum())
            # Preserve non-estimability rather than quietly dropping folds.
            mean=float(v.mean()) if n==5 else np.nan;sd=float(v.std(ddof=1)) if n==5 else np.nan
            row.update({metric+'_mean':mean,metric+'_sd':sd,metric+'_n_folds':n})
            display[metric]=f'{mean:.3f} ± {sd:.3f}' if n==5 else f'NA ({n}/5 estimable folds)'
        rows.append(row);show.append(display)
    full=pd.DataFrame(rows);full.to_csv(folder/f'{stem}_numeric.csv',index=False)
    pd.DataFrame(show).to_csv(folder/f'{stem}.csv',index=False,encoding='utf-8-sig')
    return full
def load_task(task):
    folder=RUN/'tasks'/task;complete=folder/'complete.json'
    assert complete.exists(),f'Incomplete task: {task}'
    done=json.loads(complete.read_text());assert all(sha(folder/n)==h for n,h in done['outputs'].items())
    # Preserve binary64 probabilities exactly, including tied RF scores, on CSV reload.
    frame=pd.read_csv(folder/'predictions.csv',float_precision='round_trip');rec=json.loads((folder/'metrics.json').read_text())
    assert frame.accession.is_unique
    assert np.array_equal(frame[LABELS].to_numpy(),master_truth().loc[frame.accession,LABELS].to_numpy()),'Prediction labels do not match source master'
    y=frame[LABELS].to_numpy();p=frame[SCORES].to_numpy()
    comp=module('14_mrmr_model_comparison.py','cv_metric_check')
    calculated=comp.metrics(y,p,np.array(rec['thresholds']))
    for key,value in calculated.items():assert np.isclose(value,rec['metrics'][key],rtol=0,atol=1e-12),(task,key)
    f=int(task.split('__')[2]);design=task.split('__')[1]
    splits=pd.read_csv(RUN/'splits'/f'{design}.csv')
    assert set(frame.accession)==set(splits.loc[splits.outer_fold.eq(f),'accession'])
    return frame,rec
def load_five(prefix,suffix=''):
    loaded=[load_task(f'{prefix}{f}{suffix}') for f in range(5)]
    merged=pd.concat([p.assign(fold=f) for f,(p,_) in enumerate(loaded)],ignore_index=True)
    assert merged.accession.is_unique
    assert set(merged.accession)==set(pd.read_csv(RUN/'splits/random.csv').accession)
    return loaded,merged
def curve(frame,c=None):
    y=frame[LABELS].to_numpy();p=frame[SCORES].to_numpy()
    ids=range(7) if c is None else [c]
    curves=[]
    for j in ids:
        f,t,_=roc_curve(y[:,j],p[:,j]);curves.append(np.interp(GRID,f,t))
    v=np.mean(curves,axis=0);v[0]=0.;v[-1]=1.;return v
def roc(ax,frames,label,color,c=None,ls='-'):
    arr=np.array([curve(p,c) for p in frames]);means=arr.mean(axis=0);sd=arr.std(axis=0,ddof=1)
    aucs=np.array([roc_auc_score(p[LABELS],p[SCORES],average='macro') if c is None else roc_auc_score(p[LABELS[c]],p[SCORES[c]]) for p in frames])
    ax.plot(GRID,means,color=color,ls=ls,lw=2,label=f'{label}: {aucs.mean():.3f} ± {aucs.std(ddof=1):.3f}')
    ax.fill_between(GRID,np.maximum(0,means-sd),np.minimum(1,means+sd),color=color,alpha=.1)
def roc_axes(ax):
    ax.plot([0,1],[0,1],':',color='#777777',lw=1)
    ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xlabel='False positive rate',ylabel='True positive rate')
    ax.legend(loc='lower right',frameon=True,fontsize=17)
    ax.spines[['top','right']].set_visible(False)
def finish(number,details):
    folder=out(number);dump(folder/'five_fold_provenance.json',{'figure':number,'status':'staged',
        'source_run':str(RUN.relative_to(ROOT)),'outer_folds':5,'sd_ddof':1,'display_decimals':3,**details})

def dataset():
    folder=out(2);source=original_section(2)
    # These four panels describe the same entire dataset, not model performance.
    for p in source.iterdir():
        if p.is_file() and p.suffix=='.csv':
            table=pd.read_csv(p)
            table.to_csv(folder/p.name,index=False,float_format='%.3f')
            shutil.copy2(p,folder/(p.stem+'_full_precision.csv'))
    stems=['figure_2A_class_distribution','figure_2B_length_distribution','figure_2C_taxonomic_distribution','figure_2D_label_combinations']
    for s in stems:
        for ext in ['svg','pdf','png']:shutil.copy2(source/f'{s}.{ext}',folder/f'{s}.{ext}')
    # Keep existing black median and user-approved typography; composition rerun.
    if (source/'source_manifest.json').exists():shutil.copy2(source/'source_manifest.json',folder/'source_manifest.json')
    compose(folder,2,stems)
    shutil.copy2(RUN/'fold_sample_counts.csv',folder/'five_fold_sample_counts.csv')
    finish(2,{'descriptive_only':True,'error_bars':'not applicable to population counts','source_panels_unchanged':True})

def comparison():
    folder=out(3);foldrows=[];perclass=[];frames={};select=[];ifs=[]
    for name in NAMES:
        tag=name.replace(' ','_');loaded,oof=load_five('selected__random__',f'__{tag}')
        oof.to_csv(folder/f'oof_{tag}.csv',index=False);frames[name]=[p for p,r in loaded]
        for f,(p,r) in enumerate(loaded):
            foldrows.append({'model':name,'fold':f,**r['metrics']})
            select.append({'model':name,'fold':f,**r['selection']})
            for c in range(7):perclass.append({'model':name,'fold':f,'ec_class':c+1,'class_name':CLASSES[c],
                'roc_auc':roc_auc_score(p[LABELS[c]],p[SCORES[c]]),'ap':average_precision_score(p[LABELS[c]],p[SCORES[c]])})
            for k in ([0] if name=='1D-CNN' else [16,32,64,128,256,432]):
                rec=json.loads((RUN/'tasks'/f'compare__random__{f}__{tag}__{k}'/'validation.json').read_text());ifs.append(rec)
    scores=summary(pd.DataFrame(foldrows),['model'],METRICS,folder,'algorithm_test_metrics').set_index('model')
    summary(pd.DataFrame(perclass),['model','ec_class','class_name'],['roc_auc','ap'],folder,'per_class_metrics')
    pd.DataFrame(select).to_csv(folder/'selected_features_by_fold.csv',index=False)
    curves=summary(pd.DataFrame(ifs),['model','n_features'],['validation_macro_auc'],folder,'ifs_validation_results')
    style();fig,ax=plt.subplots(figsize=(18,8));xs=np.arange(8);w=.085
    for i,name in enumerate(NAMES):
        means=[scores.loc[name,k+'_mean'] for k in METRICS];sd=[scores.loc[name,k+'_sd'] for k in METRICS]
        ax.bar(xs+(i-3)*w,means,w*.9,yerr=sd,color=COLORS[i],alpha=.5,capsize=2,edgecolor='#555555',lw=.5,label=name)
    ax.set_xticks(xs,['ROC-AUC','AP','Accuracy\n(exact match)','Precision','Recall','F1','Specificity','MCC'])
    ax.set(ylim=(0,1.07),ylabel='Five-fold test score (mean ± SD)');ax.legend(ncol=7,loc='upper center',bbox_to_anchor=(.5,1.15),fontsize=16)
    save(fig,folder,'figure_3A_algorithm_metrics')
    fig,ax=plt.subplots(figsize=(16,8))
    for i,name in enumerate(NAMES):roc(ax,frames[name],name,COLORS[i],ls=['-','--','-.',':'][i%4])
    roc_axes(ax);save(fig,folder,'figure_3B_algorithm_macro_roc')
    fig,ax=plt.subplots(figsize=(16,8))
    for i,name in enumerate(NAMES[:-1]):
        sub=curves[curves.model.eq(name)].sort_values('n_features');mean=sub.validation_macro_auc_mean.to_numpy();sd=sub.validation_macro_auc_sd.to_numpy();x=sub.n_features.to_numpy()
        best=int(np.argmax(mean));ax.plot(x,mean,marker=['o','s','^','D','v','P'][i],color=COLORS[i],label=f'{name} [{x[best]}, {mean[best]:.3f} ± {sd[best]:.3f}]')
        ax.fill_between(x,mean-sd,mean+sd,color=COLORS[i],alpha=.12)
    ax.set(xlabel='Number of features in inner-training mRMR ranking',ylabel='Inner-validation macro ROC-AUC (mean ± SD)')
    ax.legend(loc='lower right',frameon=True,fontsize=17);save(fig,folder,'figure_3C_mrmr_ifs')
    fig,ax=plt.subplots(figsize=(16,8))
    for c in range(7):roc(ax,frames['LightGBM'],f'EC{c+1} {CLASSES[c]}',COLORS[c],c)
    roc_axes(ax);save(fig,folder,'figure_3D_per_class_roc')
    compose(folder,3,['figure_3A_algorithm_metrics','figure_3B_algorithm_macro_roc','figure_3C_mrmr_ifs','figure_3D_per_class_roc'])
    finish(3,{'model':'comparison configuration; Figure 3D LightGBM','ifs':'inner validation, not outer test','cnn_ifs':'not applicable: raw sequence branch'})

def generalization():
    folder=out(4);allframes={};rows=[];pc=[];aux=[]
    for design in DESIGNS:
        loaded,oof=load_five(f'lgb__{design}__');allframes[design]=[p for p,r in loaded];oof.to_csv(folder/f'oof_{design}.csv',index=False)
        for f,(p,r) in enumerate(loaded):
            rows.append({'evaluation':design,'fold':f,**r['metrics']})
            aux.append({'evaluation':design,'fold':f,'roc_auc':r['metrics']['auxiliary_roc_auc'],'ap':r['metrics']['auxiliary_ap']})
            for c in range(7):pc.append({'evaluation':design,'fold':f,'ec_class':c+1,'class_name':CLASSES[c],
                                      'roc_auc':roc_auc_score(p[LABELS[c]],p[SCORES[c]]),'ap':average_precision_score(p[LABELS[c]],p[SCORES[c]])})
    scores=summary(pd.DataFrame(rows),['evaluation'],METRICS+['micro_roc_auc'],folder,'homology_auc_summary').set_index('evaluation')
    pcs=summary(pd.DataFrame(pc),['evaluation','ec_class','class_name'],['roc_auc','ap'],folder,'homology_per_class_auc')
    summary(pd.DataFrame(aux),['evaluation'],['roc_auc','ap'],folder,'multifunction_auxiliary_metrics')
    style();fig,ax=plt.subplots(figsize=(13,7));sp=pcs[pcs.evaluation.eq('species')]
    ax.bar(range(7),sp.roc_auc_mean,yerr=sp.roc_auc_sd,color=COLORS,alpha=.5,capsize=5)
    ax.set_xticks(range(7),[f'EC{i}' for i in range(1,8)]);ax.set(ylim=(0,1.12),ylabel='Species-disjoint ROC-AUC (mean ± SD)')
    for i,r in enumerate(sp.itertuples()):ax.text(i,r.roc_auc_mean+r.roc_auc_sd+.025,f'{r.roc_auc_mean:.3f}\n± {r.roc_auc_sd:.3f}',ha='center',fontsize=14)
    save(fig,folder,'figure_4A_species_per_class_auc')
    fig,ax=plt.subplots(figsize=(13,8))
    for c in range(7):roc(ax,allframes['species'],f'EC{c+1} {CLASSES[c]}',COLORS[c],c)
    roc_axes(ax);save(fig,folder,'figure_4B_species_roc_curves')
    fig,ax=plt.subplots(figsize=(14,8));thresholds=list(range(90,0,-10))
    for key,color,delta in [('ROC-AUC',COLORS[1],.025),('micro_roc_auc',COLORS[5],-.045)]:
        means=np.array([scores.loc[f'homology_{t}',key+'_mean'] for t in thresholds]);sd=np.array([scores.loc[f'homology_{t}',key+'_sd'] for t in thresholds])
        ax.errorbar(thresholds,means,yerr=sd,color=color,marker='o',capsize=4,label='Macro AUC' if key=='ROC-AUC' else 'Micro AUC')
        for t,m,s in zip(thresholds,means,sd):ax.annotate(f'{m:.3f}\n± {s:.3f}',(t,m),xytext=(0,25 if delta>0 else -44),textcoords='offset points',ha='center',fontsize=14,color=color)
    ax.axhline(.8,ls=':',color='#777777');ax.set_xticks(thresholds);ax.set(xlim=(95,5),ylim=(0,1.16),xlabel='Homology identity threshold (%)',ylabel='Five-fold ROC-AUC (mean ± SD)')
    ax.legend(loc='lower left');save(fig,folder,'figure_4C_homology_auc_ladder')
    fig,axes=plt.subplots(3,3,figsize=(20,20))
    for ax,t in zip(axes.flat,thresholds):
        fs=allframes[f'homology_{t}']
        for c in range(7):roc(ax,fs,f'EC{c+1}',COLORS[c],c)
        roc(ax,fs,'Macro','#222222',ls='--');roc_axes(ax);ax.set_title(f'{t}% identity');ax.legend(loc='lower right',fontsize=16)
    save(fig,folder,'figure_4D_homology_roc_curves')
    fig,ax=plt.subplots(figsize=(15,12));mat=[];ann=[]
    for t in thresholds:
        sub=pcs[pcs.evaluation.eq(f'homology_{t}')].sort_values('ec_class');mat.append(sub.roc_auc_mean.tolist());ann.append([f'{r.roc_auc_mean:.3f}\n± {r.roc_auc_sd:.3f}' for r in sub.itertuples()])
    sns.heatmap(np.array(mat),annot=np.array(ann),fmt='',cmap=REFERENCE_SEQUENTIAL_CMAP,vmin=0,vmax=1,linewidths=.8,xticklabels=[f'EC{i}' for i in range(1,8)],yticklabels=[f'{t}%' for t in thresholds],ax=ax,annot_kws={'size':14},cbar_kws={'label':'Mean ROC-AUC'})
    ax.set(xlabel='Enzyme class',ylabel='Homology identity threshold');save(fig,folder,'figure_4E_homology_per_class_auc')
    compose(folder,4,['figure_4A_species_per_class_auc','figure_4B_species_roc_curves','figure_4C_homology_auc_ladder','figure_4D_homology_roc_curves','figure_4E_homology_per_class_auc'])
    finish(4,{'model':'historical downstream LightGBM configuration, newly trained per fold','curves':'mean TPR ± fold SD; legend mean fold AUC ± SD'})

def ablation():
    folder=out(5);tags=['full','aac','dpc','phys','aac_dpc','aac_phys','dpc_phys'];names=['Full representation','AAC','DPC','Physicochemical','AAC + DPC','AAC + Physicochemical','DPC + Physicochemical']
    rows=[];pc=[]
    for tag,name in zip(tags,names):
        loaded,oof=load_five('lgb__random__' if tag=='full' else 'ablation__random__','' if tag=='full' else f'__{tag}')
        oof.to_csv(folder/f'oof_{tag}.csv',index=False)
        for f,(p,r) in enumerate(loaded):
            rows.append({'tag':tag,'feature_set':name,'fold':f,**r['metrics']})
            for c in range(7):pc.append({'tag':tag,'ec_class':c+1,'fold':f,'roc_auc':roc_auc_score(p[LABELS[c]],p[SCORES[c]]),'ap':average_precision_score(p[LABELS[c]],p[SCORES[c]])})
    s=summary(pd.DataFrame(rows),['tag','feature_set'],METRICS,folder,'feature_ablation_auc_ap').set_index('tag')
    summary(pd.DataFrame(pc),['tag','ec_class'],['roc_auc','ap'],folder,'per_class_ablation_metrics')
    shutil.copy2(folder/'feature_ablation_auc_ap.csv',folder/'feature_ablation_auc.csv')
    style();cols=['#555555','#5B6F82','#6D91B3','#819DA5','#C38B58','#AB263A','#CE7D77']
    for key,stem,yl in [('ROC-AUC','figure_5A_feature_ablation_auc','Macro ROC-AUC'),('AP','figure_5B_feature_ablation_ap','Macro average precision (AP)')]:
        fig,ax=plt.subplots(figsize=(17,7));mean=np.array([s.loc[t,key+'_mean'] for t in tags]);sd=np.array([s.loc[t,key+'_sd'] for t in tags]);low=max(0,float((mean-sd).min())-.07);high=1.0
        for start,end,shade,label in [(-.6,.5,'#F2F2F2','Full representation'),(.5,3.5,'#E8EFF6','Single feature group'),(3.5,6.6,'#F5E9DF','Paired feature groups')]:
            ax.axvspan(start,end,color=shade,alpha=.5);ax.text((start+end)/2,high-.015,label,ha='center',va='top',fontsize=16)
        for i,(m,sdev,col) in enumerate(zip(mean,sd,cols)):
            ax.vlines(i,low,m,color=col,alpha=.5,lw=2);ax.errorbar(i,m,yerr=sdev,color=col,capsize=5,alpha=.5)
            ax.scatter(i,m,s=140,facecolors='none',edgecolors=col,alpha=.5);ax.scatter(i,m,s=35,c=col,alpha=.5)
            ax.text(i,m+sdev+.012,f'{m:.3f} ± {sdev:.3f}',ha='center',fontsize=14)
        ax.set_xticks(range(7),['Full\n(432)','AAC\n(20)','DPC\n(400)','Physicochemical\n(12)','AAC + DPC\n(420)','AAC +\nPhysicochemical\n(32)','DPC +\nPhysicochemical\n(412)'])
        ax.set(xlim=(-.6,6.6),ylim=(low,high),ylabel=yl+' (mean ± SD)');save(fig,folder,stem)
    compose(folder,5,['figure_5A_feature_ablation_auc','figure_5B_feature_ablation_ap'],[[('figure_5A_feature_ablation_auc',1)],[('figure_5B_feature_ablation_ap',1)]])
    supplementary_multifunction(folder)
    finish(5,{'paired_folds':True,'same_model_seed_within_fold':True,'axis':'truncated, inspect values and SD, stem length is not a ratio effect'})

def supplementary_multifunction(folder):
    source=module('12_run_multifunction_interpretability.py','cv_auxiliary_report')
    comp=module('14_mrmr_model_comparison.py','cv_source_compare')
    data=pd.read_csv(ROOT/'data/processed/archaea_swissprot_ec_master.csv');x=np.load(RUN/'features.npy')
    split=pd.read_csv(RUN/'splits/random.csv');decode=[];thresholds=[];annotation=[];aux=[];gain=[];examples=[]
    for f in range(5):
        p,rec=load_task(f'lgb__random__{f}');models=RUN/'tasks'/f'lgb__random__{f}'
        va=split[f'role_fold_{f}'].eq('validation').to_numpy();test=data.set_index('accession').loc[p.accession].reset_index()
        truth=p[LABELS].to_numpy();scores=p[SCORES].to_numpy();multi=p.label_aux.to_numpy();auxscore=p.score_aux.to_numpy()
        valp=np.column_stack([joblib.load(models/f'ec{c+1}.joblib').predict_proba(x[va])[:,1] for c in range(7)])
        valaux=joblib.load(models/'ec8.joblib').predict_proba(x[va])[:,1]
        grid=np.linspace(.05,.95,181);yv=data.loc[va,LABELS].to_numpy();ym=data.loc[va,'is_multifunctional_top_level'].astype(int).to_numpy()
        cut=np.array([grid[np.argmax([f1_score(yv[:,c],valp[:,c]>=v,zero_division=0) for v in grid])] for c in range(7)])
        acut=grid[np.argmax([f1_score(ym,valaux>=v,zero_division=0) for v in grid])]
        for c in range(7):thresholds.append({'fold':f,'ec_class':c+1,'validation_f1_threshold':cut[c]})
        fixed=source.force_nonempty(scores>=.5,scores);perclass=source.force_nonempty(scores>=cut,scores);guided=np.zeros_like(perclass)
        for i in range(len(p)):
            if auxscore[i]>=acut:
                guided[i]=perclass[i]
                if guided[i].sum()<2:guided[i,np.argsort(scores[i])[-2:]]=1
            else:guided[i,np.argmax(scores[i])]=1
        for name,pred in [('Fixed 0.50',fixed),('Validation class-specific',perclass),('Auxiliary-guided adaptive',guided)]:decode.append({'fold':f,**source.decoding_metrics(name,truth,pred)})
        aux.append({'fold':f,'model':'multifunction LightGBM','roc_auc':roc_auc_score(multi,auxscore),'ap':average_precision_score(multi,auxscore),'validation_threshold':acut})
        ann=source.annotation_auc_table(test,truth,scores);ann['fold']=f
        ann['evaluable_class_mean_auc']=ann.macro_roc_auc
        ann.loc[ann.evaluable_classes.ne(7),'macro_roc_auc']=np.nan
        annotation.extend(ann.to_dict('records'))
        gm=[]
        for c in range(7):
            m=joblib.load(models/f'ec{c+1}.joblib');v=m.booster_.feature_importance(importance_type='gain');gm.append(v/max(v.sum(),1))
        for feature,value in zip(comp.FEATURES,np.mean(gm,axis=0)):gain.append({'fold':f,'feature':feature,'mean_normalized_gain':value})
        loss=-(truth*np.log(np.clip(scores,1e-7,1))+(1-truth)*np.log(np.clip(1-scores,1e-7,1))).mean(axis=1)
        exact=np.all(truth==guided,axis=1)
        for label,mask,reverse in [('High-confidence correct',exact,False),('High-loss incorrect',~exact,True),('Correct multifunctional',exact&(multi==1),False)]:
            eligible=np.flatnonzero(mask);order=eligible[np.argsort(loss[eligible])];order=order[::-1] if reverse else order
            for i in order[:5]:examples.append({'fold':f,'category':label,'accession':p.accession.iloc[i],
                'true_classes':';'.join(str(c+1) for c in np.flatnonzero(truth[i])),
                'predicted_classes':';'.join(str(c+1) for c in np.flatnonzero(guided[i])),
                'multifunction_probability':auxscore[i],'mean_binary_cross_entropy':loss[i]})
    summary(pd.DataFrame(decode),['strategy'],['macro_f1','micro_f1','sample_f1','exact_match','hamming_loss'],folder,'decoding_comparison')
    summary(pd.DataFrame(thresholds),['ec_class'],['validation_f1_threshold'],folder,'adaptive_thresholds')
    summary(pd.DataFrame(aux),['model'],['roc_auc','ap','validation_threshold'],folder,'multifunction_auxiliary_metrics')
    summary(pd.DataFrame(annotation),['annotation','status'],['macro_roc_auc'],folder,'annotation_stratified_auc')
    summary(pd.DataFrame(gain),['feature'],['mean_normalized_gain'],folder,'feature_gain_importance')
    pd.DataFrame(examples).to_csv(folder/'representative_proteins.csv',index=False,float_format='%.3f')
    pd.DataFrame(examples).to_csv(folder/'representative_proteins_full_precision.csv',index=False)

def interpretability():
    folder=out(6);source=original_section(6);interp=module('13_run_seven_class_interpretability.py','cv_interpret')
    data=pd.read_csv(ROOT/'data/processed/archaea_swissprot_ec_master.csv');rows=[]
    loaded,oof=load_five('lgb__homology_40__')
    for f,(p,_) in enumerate(loaded):
        frame=p.merge(data[['accession','n_top_classes','fragment']],on='accession',validate='one_to_one')
        frame=frame[frame.n_top_classes.eq(1)&frame.fragment.fillna('').ne('fragment')]
        for a,b in combinations(range(7),2):
            pair=frame[frame[LABELS[a]].eq(1)|frame[LABELS[b]].eq(1)];truth=pair[LABELS[a]].to_numpy()
            left=np.clip(pair[SCORES[a]],1e-7,1-1e-7);right=np.clip(pair[SCORES[b]],1e-7,1-1e-7)
            score=np.log(left/(1-left))-np.log(right/(1-right))
            val=roc_auc_score(truth,score) if len(np.unique(truth))==2 else np.nan
            rows.append({'class_a':a+1,'class_b':b+1,'fold':f,'pairwise_roc_auc':val,'n_a':int(truth.sum()),'n_b':int(len(truth)-truth.sum())})
    pairwise=summary(pd.DataFrame(rows),['class_a','class_b'],['pairwise_roc_auc'],folder,'pairwise_class_auc')
    rows=[];feature_names=[f'AAC_{a}' for a in interp.AMINO_ACIDS]+[f'DPC_{a}{b}' for a in interp.AMINO_ACIDS for b in interp.AMINO_ACIDS]+[n.replace(' ','_') for n in interp.PHYS_NAMES]
    for f in range(5):
        d=RUN/'tasks'/f'shap__random__{f}';assert (d/'complete.json').exists()
        vals=np.load(d/'shap_summary.npy')
        for c in range(7):
            for j,name in enumerate(feature_names):rows.append({'fold':f,'ec_class':c+1,'class_name':CLASSES[c],'feature':name,'feature_group':'AAC' if j<20 else ('DPC' if j<420 else 'Physicochemical'),'mean_absolute_shap':vals[c,j,0],'positive_negative_contrast':vals[c,j,1]})
    shap=summary(pd.DataFrame(rows),['ec_class','class_name','feature','feature_group'],['mean_absolute_shap','positive_negative_contrast'],folder,'class_specific_shap')
    grouped=pd.DataFrame(rows).groupby(['fold','ec_class','class_name','feature_group'],as_index=False).mean_absolute_shap.sum()
    grouped['group_share']=grouped.mean_absolute_shap/grouped.groupby(['fold','ec_class']).mean_absolute_shap.transform('sum')
    summary(grouped,['ec_class','class_name','feature_group'],['mean_absolute_shap','group_share'],folder,'class_shap_group_summary')
    examples=[]
    example_frame=oof.merge(data[['accession','protein_names','organism','fragment','n_top_classes']],on='accession',validate='one_to_one')
    example_frame=example_frame[example_frame.n_top_classes.eq(1)&example_frame.fragment.fillna('').ne('fragment')]
    for c in range(7):
        positives=example_frame[example_frame[LABELS[c]].eq(1)].sort_values(SCORES[c],ascending=False)
        for selection,subset in [('Highest-score annotated positive',positives.head(3)),('Lowest-score annotated positive',positives.tail(1))]:
            for rank,(_,r) in enumerate(subset.iterrows(),1):examples.append({'ec_class':c+1,'class_name':CLASSES[c],'selection':selection,'rank':rank,'fold':r.fold,'accession':r.accession,'protein_names':r.protein_names,'organism':r.organism,'class_probability':r[SCORES[c]]})
    pd.DataFrame(examples).to_csv(folder/'representative_class_proteins.csv',index=False,float_format='%.3f')
    pd.DataFrame(examples).to_csv(folder/'representative_class_proteins_full_precision.csv',index=False)
    selected=[]
    for c in range(1,8):selected+=shap[shap.ec_class.eq(c)].nlargest(3,'mean_absolute_shap_mean').feature.tolist()
    selected=list(dict.fromkeys(selected));shapfold=pd.DataFrame(rows)
    # Per-fold normalization, then mean and sample SD (not SD on a pooled model).
    normrows=[]
    for (f,c),g in shapfold[shapfold.feature.isin(selected)].groupby(['fold','ec_class']):
        scale=g.positive_negative_contrast.abs().max() or 1
        for r in g.itertuples():normrows.append({'fold':f,'ec_class':c,'feature':r.feature,'normalized_contrast':r.positive_negative_contrast/scale})
    norm=summary(pd.DataFrame(normrows),['ec_class','feature'],['normalized_contrast'],folder,'normalized_shap_contrast')
    for name in ['class_descriptive_statistics.csv','class_taxonomic_breadth.csv','class_physicochemical_effects.csv','multifunction_cooccurrence.csv']:
        frame=pd.read_csv(source/name);frame.to_csv(folder/name,index=False,float_format='%.3f')
        shutil.copy2(source/name,folder/(Path(name).stem+'_full_precision.csv'))
    # Descriptive panels are preserved; they do not have a CV sampling unit.
    for stem in ['figure_6A_class_property_heatmap','figure_6D_multifunction_network']:
        for ext in ['svg','pdf','png']:shutil.copy2(source/f'{stem}.{ext}',folder/f'{stem}.{ext}')
    style();matrix=np.full((7,7),np.nan);ann=np.full((7,7),'—',dtype=object)
    for r in pairwise.itertuples():
        a,b=r.class_a-1,r.class_b-1;matrix[a,b]=matrix[b,a]=r.pairwise_roc_auc_mean
        ann[a,b]=ann[b,a]=f'{r.pairwise_roc_auc_mean:.3f}\n± {r.pairwise_roc_auc_sd:.3f}' if np.isfinite(r.pairwise_roc_auc_mean) else 'NA'
    fig,ax=plt.subplots(figsize=(13,11));sns.heatmap(matrix,annot=ann,fmt='',mask=np.eye(7,dtype=bool),vmin=0,vmax=1,cmap=REFERENCE_SEQUENTIAL_CMAP,linewidths=1,xticklabels=[f'EC{i}' for i in range(1,8)],yticklabels=[f'EC{i}' for i in range(1,8)],annot_kws={'size':16},cbar_kws={'label':'Pairwise ROC-AUC (mean)'},ax=ax)
    ax.set(title='Homology-40: five-fold pairwise discrimination',xlabel='Enzyme class',ylabel='Enzyme class');save(fig,folder,'figure_6B_pairwise_auc_heatmap')
    mat=norm.pivot(index='ec_class',columns='feature',values='normalized_contrast_mean').loc[range(1,8),selected]
    sd=norm.pivot(index='ec_class',columns='feature',values='normalized_contrast_sd').loc[range(1,8),selected]
    annot=np.array([[f'{m:.3f}\n± {s:.3f}' for m,s in zip(row,srow)] for row,srow in zip(mat.values,sd.values)])
    fig,ax=plt.subplots(figsize=(max(18,len(selected)*1.25),10));sns.heatmap(mat,annot=annot,fmt='',annot_kws={'size':15},cmap=REFERENCE_DIVERGING_CMAP,center=0,vmin=-1,vmax=1,linewidths=.5,cbar_kws={'label':'Normalized SHAP contrast (mean)'},ax=ax)
    ax.tick_params(axis='both',labelsize=24)
    ax.set(xlabel='Predictive feature',ylabel='EC class',title='Outer-test TreeSHAP: five-fold mean ± SD');ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha='right')
    save(fig,folder,'figure_6C_class_shap_heatmap')
    compose(folder,6,[],[[('figure_6A_class_property_heatmap',1.5),('figure_6B_pairwise_auc_heatmap',1)],
                        [('figure_6C_class_shap_heatmap',1.5),('figure_6D_multifunction_network',1)]])
    finish(6,{'pairwise_population':'complete single-label, Homology-40 outer tests','shap_population':'complete single-label, random outer tests',
              'descriptive_panels':['6A','6D'],'diagonal':'not a discrimination task; masked, not a fabricated AUC=1'})

def validate():
    report={'status':'verified','figures':[],'display':'mean ± sample SD, ddof=1, 3 decimals','raw_precision':'preserved'}
    import xml.etree.ElementTree as ET
    for number in range(2,7):
        folder=out(number);assert (folder/'five_fold_provenance.json').exists()
        path=folder/f'figure_{number}_combined.svg';root=ET.parse(path).getroot();ns='{http://www.w3.org/2000/svg}'
        assert not list(root.iter(ns+'image'))
        ids=[e.get('id') for e in root.iter() if e.get('id')];assert len(ids)==len(set(ids))
        report['figures'].append({'number':number,'svg_sha256':sha(path),'path':str(path.relative_to(ROOT))})
    checked=0
    for numeric in STAGE.glob('*/*_numeric.csv'):
        stem=numeric.name.removesuffix('_numeric.csv');raw=pd.read_csv(numeric.parent/f'{stem}_folds.csv');num=pd.read_csv(numeric);display=pd.read_csv(numeric.parent/f'{stem}.csv')
        metrics=[k.removesuffix('_mean') for k in num.columns if k.endswith('_mean')]
        groups=[k for k in num.columns if not any(k==m+s for m in metrics for s in ['_mean','_sd','_n_folds'])]
        assert len(num)==len(display)
        for i,row in num.iterrows():
            block=raw
            for g in groups:block=block.loc[block[g].eq(row[g])]
            assert len(block)==5 and set(block.fold)==set(range(5))
            for metric in metrics:
                values=block[metric].to_numpy(float);n=np.isfinite(values).sum()
                assert row[metric+'_n_folds']==n
                if n==5:
                    assert np.isclose(row[metric+'_mean'],values.mean(),rtol=0,atol=1e-12)
                    assert np.isclose(row[metric+'_sd'],values.std(ddof=1),rtol=0,atol=1e-12)
                    assert display.iloc[i][metric]==f'{values.mean():.3f} ± {values.std(ddof=1):.3f}'
                else:assert pd.isna(row[metric+'_mean']) and str(display.iloc[i][metric]).startswith('NA')
                checked+=1
    report['summary_cells_independently_recomputed']=checked
    report['statistical_interpretation']={'sd':'sample SD of 5 outer-test fold metrics; not SEM or CI',
        'fold_independence':'training sets overlap; no independence-based significance claim',
        'accuracy':'multilabel exact-match accuracy','AP':'mean of seven average_precision_score values, not trapezoidal PR area',
        'external_confirmation':False,'causality':'SHAP and co-occurrence are predictive/descriptive associations only'}
    dump(RUN/'report_validation.json',report)
    (RUN/'statistical_validation_notes.md').write_text('''# 五折结果统计口径核查

## Material Passport

状态：RECOMPUTED。预测文件与主数据标签对应关系、五折测试覆盖、逐折八项指标已按报告脚本重新核算；不声称进行了独立重训练复现。

所有性能显示均值 ± 样本标准差（ddof=1），保留三位小数。标准差是折间离散程度，不是标准误或95%置信区间。五折训练集存在重叠，不据小幅差异或SD重叠与否宣称统计显著性。宏AUC为每折七个OvR AUC算术平均，再计算五折均值；不是汇集OOF预测后计算的一次AUC。

## 统计与方法风险清单（11/11覆盖）

1. Simpson悖论：保留宏/微AUC及逐类别结果，未把总体趋势等同于所有类别趋势；未逐物种检验反转。
2. 生态谬误：样本级预测评估与物种/同源簇分组分开报告，不由组平均外推单个蛋白机制。
3. 选择偏倚：Swiss-Prot reviewed且具有EC注释的古菌蛋白是经筛选总体；不代表全部未知古菌蛋白。
4. 碰撞变量偏倚：未开展因果调整模型；注释条件与单功能亚组分析作为描述性敏感性分析。
5. 基率忽略：保存各折正例数、AP与ROC-AUC；类别不平衡影响不能仅凭AUC排除。
6. 均值回归：不是按极端分数选择的前后干预研究；不以单次最好结果作为基准提升依据。
7. 幸存者偏倚：只在五折齐全后发布主结果，失败任务不静默丢弃；不可估计单元记录NA。
8. 多重寻找：保留全部七算法、所有特征前缀和九个同源阈值，没有新增未校正显著性检验。
9. 分析路径自由度：该数据已在早期开发中多次查看；新五折不是独立外部确认。配置与选择规则留痕。
10. 相关不等于因果：SHAP与标签共现不证明催化机制；消融仅描述当前预测流程的变化。
11. 反向因果：未提出因果方向或实验干预结论，不把预测能力解释成因果效应。

图2和图6的纯描述性面板没有人为增加折间SD。图6的成对分类对角线不构成分类任务，已遮罩；缺失类别的折不能用AUC=0或1补齐。
''',encoding='utf-8')
    return report

def publish():
    validate();backup=ROOT/'results/_archive_pre_five_fold_20261007';manifest=[];retired=[]
    for number in range(2,7):
        folder=out(number);dest=ROOT/'results'/SECTIONS[number];archive=backup/SECTIONS[number];archive.mkdir(parents=True,exist_ok=True)
        # Snapshot every existing top-level artifact, including tables not replaced.
        for p in dest.iterdir():
            if p.is_file() and not (archive/p.name).exists():shutil.copy2(p,archive/p.name)
        incoming={p.name for p in folder.iterdir() if p.is_file()}
        legacy=dest/'legacy_single_split'
        for p in list(dest.iterdir()):
            if p.is_file() and p.suffix in ['.csv','.json','.md'] and p.name not in incoming and p.name!='FIVE_FOLD_RESULTS_README.md':
                assert p.resolve().is_relative_to(ROOT.resolve()) and legacy.resolve().is_relative_to(ROOT.resolve())
                legacy.mkdir(exist_ok=True);target=legacy/p.name
                if target.exists():raise FileExistsError(f'Refuse to overwrite historical artifact: {target}')
                assert sha(p)==sha(archive/p.name)
                shutil.move(str(p),str(target));retired.append(str(target.relative_to(ROOT)))
        for p in folder.iterdir():
            if p.is_file():shutil.copy2(p,dest/p.name);manifest.append({'path':str((dest/p.name).relative_to(ROOT)),'sha256':sha(dest/p.name)})
        (dest/'FIVE_FOLD_RESULTS_README.md').write_text('本节当前图表来自 results/five_fold_cv_20261007。性能展示为外层五折均值 ± 样本标准差（ddof=1），保留三位小数。*_folds.csv与*_numeric.csv保留全精度。描述性计数没有人为添加标准差。旧单次实验文件已备份在 results/_archive_pre_five_fold_20261007；未更新的旧顶层数据表、报告和草稿归入本节 legacy_single_split 子目录，不属于五折结果。旧模型/阈值子目录仅作历史追溯，新模型均位于five_fold_cv_20261007/tasks。重绘请用scripts/27_report_five_fold.py，旧单次绘图脚本不适用于新汇总表结构。\n',encoding='utf-8')
    dump(RUN/'published_manifest.json',{'status':'published','files':manifest,'backup':str(backup.relative_to(ROOT)),'retired_single_split_artifacts':retired})
    previous=ROOT/'docs/final_figures_manifest_20260925.json'
    if previous.exists():
        saved=backup/'docs'/previous.name;saved.parent.mkdir(parents=True,exist_ok=True)
        if not saved.exists():shutil.copy2(previous,saved)
        record=json.loads(previous.read_text(encoding='utf-8'))
        record.update(date=time.strftime('%Y-%m-%d'),status='five-fold update; numerically verified; not yet newly user-approved',training_performed=True,
                      evaluation='five outer folds, inner grouped holdout',display='mean ± sample SD (ddof=1), three decimals',source_run=str(RUN.relative_to(ROOT)))
        record.pop('figure4A_translucent_paths',None)
        for r in record['figures']:
            r['sha256']=sha(ROOT/r['path']);r['style_revision']='Five-fold update; retained Arial and approved palette/layout'
        dump(previous,record);dump(ROOT/'docs/five_fold_figures_manifest_20261007.json',record)
    experiment_report()

def experiment_report():
    algorithms=pd.read_csv(out(3)/'algorithm_test_metrics.csv')
    general=pd.read_csv(out(4)/'homology_auc_summary.csv')
    ablations=pd.read_csv(out(5)/'feature_ablation_auc_ap.csv')
    lines=['# 五折交叉验证实验更新','', '## Material Passport','',
      '状态：五折训练完成，图2–图6和相关表格已更新；原始逐折预测和模型保存在 results/five_fold_cv_20261007/tasks。',
      '任务：按用户批准方案进行外层五折、内层分组留出验证；不是内外层均五折的嵌套交叉验证。','',
      '## 评估与统计口径','',
      '随机划分以完全相同序列为组；物种划分以物种及跨物种相同序列的连通分量为组；同源划分沿用各阈值的聚类映射并约束重复序列不得跨折。每条accession在每种设计下恰好进入一次外层测试折。每个外层训练部分内部再分出一份分组验证集，训练集拟合标准化与mRMR排序，验证集负责IFS特征数、早停与分类阈值选择。',
      '每折约64%样本用于参数拟合、16%用于内部验证、20%用于外层测试，具体数量受分组约束影响。五折模型用于评估稳定性，不等于把五折模型组成部署集成；每折主任务仍为七个OvR分类器。',
      'ROC-AUC与AP按七个类别分别计算后取宏平均，随后对五个外层折计算算术均值和样本标准差（ddof=1）。所有展示值保留三位小数；原始数据不预先四舍五入。Accuracy是多标签exact-match准确率。ROC阴影为共同FPR网格上的折间TPR标准差，图例数值为原始折AUC均值与SD，两者计算对象不同。',
      '该数据在先前开发中已多次查看，新的交叉验证不构成独立外部验证；折间训练数据重叠，不据SD大小或是否重叠宣称显著性。','',
      '输入审计发现：现有10%和20%原始同源簇映射完全相同，五折划分相同，固定种子下重训结果也相同。两者保留在阈值序列中以供追溯，但不解释为独立难度或独立重复实验。','',
      '## 算法比较','', '| 算法 | Macro AUC | Macro AP |','|---|---:|---:|']
    for _,r in algorithms.iterrows():lines.append(f'| {r["model"]} | {r["ROC-AUC"]} | {r["AP"]} |')
    lines+=['','## 泛化评估','', '| 条件 | Macro AUC | Macro AP |','|---|---:|---:|']
    for _,r in general.iterrows():lines.append(f'| {r["evaluation"]} | {r["ROC-AUC"]} | {r["AP"]} |')
    lines+=['','## 特征组消融','', '| 特征组 | Macro AUC | Macro AP |','|---|---:|---:|']
    for _,r in ablations.iterrows():lines.append(f'| {r["feature_set"]} | {r["ROC-AUC"]} | {r["AP"]} |')
    lines+=['','## 图表与追溯','',
      '图2：总体数据描述，保留实际计数和长度分布，不人为增加CV标准差。图3：算法指标、宏ROC、内部验证IFS、LightGBM各类别ROC。图4：物种与10%–90%同源隔离五折。图5：七种特征组合的AUC/AP及SD。图6：描述性属性差异、Homology-40逐折成对AUC、随机五折外层测试TreeSHAP、真实标签共现。',
      'mRMR-IFS不适用于直接输入原始序列的1D-CNN，因此图3C为六条表征特征曲线，而图3A/B比较七种算法。图3采用既有算法比较配置，图4–图6采用既有下游LightGBM配置；两种配置分别重新训练，不混用模型。',
      '描述性图6A/6D保留原统计含义。SHAP解释的是预测关联，不是因果机制；成对分类对角线不计算AUC。',
      '已发布文件及SHA-256见 published_manifest.json；数值核验见 report_validation.json；划分审计见 split_audit.json、independent_input_audit.json。旧版文件在 results/_archive_pre_five_fold_20261007，可恢复。',
      '本次未自动改写整篇论文。原稿中涉及旧单次AUC、训练/验证/测试比例、CV performance或bootstrap区间的文字需要以此报告更新。']
    (RUN/'experiment_result.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['dataset','comparison','generalization','ablation','interpretability','all'],default='all');p.add_argument('--publish',action='store_true');args=p.parse_args()
    functions={'dataset':dataset,'comparison':comparison,'generalization':generalization,'ablation':ablation,'interpretability':interpretability}
    for name,fn in functions.items():
        if args.stage in ['all',name]:fn();print('STAGED',name,flush=True)
    if args.stage=='all':validate()
    if args.publish:publish()
