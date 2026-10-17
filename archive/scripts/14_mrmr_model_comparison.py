"""Train-only discrete mRMR-MID, grid IFS, and seven-model comparison.

Run from project root: python -u scripts/14_mrmr_model_comparison.py
Test predictions are generated only after validation-based selection is frozen.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef)
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
from plot_style import apply_arial_style, REFERENCE_PALETTE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/3.2_feature_selection_model_comparison'
GRID = [16, 32, 64, 128, 256, 432]
NAMES = ['Logistic Regression', 'RBF-SVM', 'Random Forest', 'XGBoost', 'LightGBM', 'DNN', '1D-CNN']
SEED = 42
AA = 'ACDEFGHIKLMNPQRSTVWY'
PHYS = ['log_length','sequence_entropy','longest_hydrophobic_run','hydrophobic_fraction',
        'charged_fraction','polar_fraction','aromatic_fraction','small_fraction',
        'glycine_fraction','proline_fraction','cysteine_fraction','histidine_fraction']
FEATURES = [f'AAC_{a}' for a in AA] + [f'DPC_{a}{b}' for a in AA for b in AA] + PHYS
apply_arial_style()
torch.set_num_threads(4)
np.random.seed(SEED)
torch.manual_seed(SEED)


def log(message):
    line = time.strftime('%H:%M:%S') + ' ' + message
    print(line, flush=True)
    with (OUT / 'training.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def discrete_mi(a, b, bins_b):
    counts = np.bincount(a.astype(np.int64) * bins_b + b, minlength=10*bins_b).reshape(10, bins_b)
    joint = counts / counts.sum()
    expected = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))


def mrmr(x, y):
    """MID = mean label MI - mean MI with already selected features.

    Train-only quantile discretization (at most ten bins); ties remain ties.
    Multi-label relevance is the unweighted mean across the seven binary labels.
    """
    edges = [np.unique(np.quantile(x[:, j], np.arange(1,10)/10)) for j in range(x.shape[1])]
    z = np.column_stack([np.searchsorted(e, x[:, j], side='left') for j,e in enumerate(edges)])
    relevance = np.array([[discrete_mi(z[:,j], y[:,c], 2) for c in range(7)] for j in range(x.shape[1])])
    mean = relevance.mean(axis=1)
    redundancy_sum = np.zeros(x.shape[1])
    chosen, rows = [], []
    for step in range(x.shape[1]):
        redundancy = redundancy_sum / max(step, 1)
        score = mean - redundancy
        score[chosen] = -np.inf
        j = int(np.argmax(score))
        rows.append({'rank':step+1,'feature_index':j,'feature':FEATURES[j],
                     'relevance':mean[j],'redundancy':redundancy[j],'mrmr_mid':score[j],
                     **{f'mi_ec{c+1}':relevance[j,c] for c in range(7)}})
        chosen.append(j)
        for other in range(x.shape[1]):
            if other not in chosen:
                redundancy_sum[other] += discrete_mi(z[:,other], z[:,j], 10)
    pd.DataFrame(rows).to_csv(OUT/'mrmr_ranking.csv', index=False)
    save_json(OUT/'mrmr_discretization.json', [e.tolist() for e in edges])
    return np.array(chosen)


class DNN(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimension,256), nn.ReLU(),nn.Dropout(0.25),
            nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.25),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,7))
    def forward(self,x):
        return self.network(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(22,16,padding_idx=0)
        self.branches = nn.ModuleList([nn.Conv1d(16,32,k,padding=k//2) for k in [3,5,7]])
        self.head = nn.Sequential(nn.Linear(96,64),nn.ReLU(),nn.Dropout(.25),nn.Linear(64,7))
    def forward(self,x):
        mask = x.ne(0).unsqueeze(1)
        e = self.embedding(x).transpose(1,2)
        pooled = [torch.relu(branch(e)).masked_fill(~mask, -1e4).amax(dim=2) for branch in self.branches]
        return self.head(torch.cat(pooled,dim=1))


def encode_sequences(sequences):
    # Fixed before evaluation: max 1024 residues, first and last 512 if longer.
    lookup = {a:i+1 for i,a in enumerate(AA)}
    arr = np.zeros((len(sequences),1024),dtype=np.int64)
    for i,s in enumerate(sequences):
        s = s if len(s)<=1024 else s[:512]+s[-512:]
        arr[i,:len(s)] = [lookup.get(a,21) for a in s]
    return arr


def neural_predict(model,x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(batch[0])).numpy() for batch in DataLoader(TensorDataset(torch.as_tensor(x)),batch_size=128)])


def train_neural(name,x,y,xv,yv):
    torch.manual_seed(SEED)
    model = CNN() if name=='1D-CNN' else DNN(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    pos_weight = torch.tensor((len(y)-y.sum(axis=0))/y.sum(axis=0),dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(TensorDataset(torch.as_tensor(x),torch.tensor(y,dtype=torch.float32)),
                        batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    best_score, stale, best_state, best_epoch = -np.inf,0,None,0
    history=[]
    for epoch in range(1,61):
        model.train()
        loss_sum=0.
        for batch,truth in loader:
            optimizer.zero_grad()
            loss=criterion(model(batch),truth)
            loss.backward()
            optimizer.step()
            loss_sum+=loss.item()*len(batch)
        score=roc_auc_score(yv,neural_predict(model,xv),average='macro')
        history.append({'epoch':epoch,'loss':loss_sum/len(y),'validation_macro_auc':score})
        if score>best_score+1e-5:
            best_score,best_state,best_epoch,stale=score,copy.deepcopy(model.state_dict()),epoch,0
        else:
            stale+=1
        if epoch%10==0:
            log(f'{name} features={x.shape[1]} epoch={epoch} validation_auc={score:.4f}')
        if stale>=8:
            break
    model.load_state_dict(best_state)
    return {'model':model,'best_epoch':best_epoch,'history':history}


def train_model(name,x,y,xv,yv):
    scaler=None
    if name in ['Logistic Regression','RBF-SVM','DNN']:
        scaler=StandardScaler().fit(x)
        x=scaler.transform(x).astype(np.float32)
        xv=scaler.transform(xv).astype(np.float32)
    if name in ['DNN','1D-CNN']:
        pack=train_neural(name,x,y,xv,yv)
    else:
        models=[]
        for c in range(7):
            yc=y[:,c]
            weight=np.where(yc==1,len(y)/(2*yc.sum()),len(y)/(2*(len(y)-yc.sum())))
            if name=='Logistic Regression':
                model=LogisticRegression(C=1,class_weight='balanced',max_iter=1500,random_state=SEED)
                model.fit(x,yc)
            elif name=='RBF-SVM':
                model=SVC(C=2,gamma='scale',class_weight='balanced',cache_size=512,random_state=SEED)
                model.fit(x,yc)
            elif name=='Random Forest':
                model=RandomForestClassifier(n_estimators=160,min_samples_leaf=2,class_weight='balanced',n_jobs=4,random_state=SEED+c)
                model.fit(x,yc)
            elif name=='XGBoost':
                model=XGBClassifier(n_estimators=800,max_depth=5,learning_rate=.05,subsample=.85,colsample_bytree=.85,
                    tree_method='hist',n_jobs=4,random_state=SEED+c,eval_metric='auc',early_stopping_rounds=40)
                model.fit(x,yc,sample_weight=weight,eval_set=[(xv,yv[:,c])],verbose=False)
            else:
                model=lgb.LGBMClassifier(n_estimators=1600,learning_rate=.03,num_leaves=31,colsample_bytree=.85,
                    reg_lambda=1,class_weight='balanced',verbosity=-1,n_jobs=4,random_state=SEED+c)
                model.fit(x,yc,eval_set=[(xv,yv[:,c])],eval_metric='auc',
                    callbacks=[lgb.early_stopping(60,verbose=False),lgb.log_evaluation(0)])
            models.append(model)
        pack={'models':models}
    pack.update({'scaler':scaler,'name':name})
    return pack


def predict(pack,x):
    if pack['scaler'] is not None:
        x=pack['scaler'].transform(x).astype(np.float32)
    if 'model' in pack:
        return neural_predict(pack['model'],x)
    # SVM sigmoid values are monotonic scores, not calibrated probabilities.
    return np.column_stack([expit(m.decision_function(x)) if pack['name']=='RBF-SVM' else m.predict_proba(x)[:,1] for m in pack['models']])


def thresholds(y,p):
    return np.array([max(np.linspace(.01,.99,99),key=lambda t:f1_score(y[:,j],p[:,j]>=t,zero_division=0)) for j in range(7)])


def metrics(y,p,cut):
    pred=p>=cut
    spec=[np.mean(~pred[y[:,j]==0,j]) for j in range(7)]
    return {'ROC-AUC':roc_auc_score(y,p,average='macro'), 'AP':average_precision_score(y,p,average='macro'),
        'Accuracy':accuracy_score(y,pred),'Precision':precision_score(y,pred,average='macro',zero_division=0),
        'Recall':recall_score(y,pred,average='macro',zero_division=0),'F1':f1_score(y,pred,average='macro',zero_division=0),
        'Specificity':np.mean(spec),'MCC':np.mean([matthews_corrcoef(y[:,j],pred[:,j]) for j in range(7)])}


def macro_curve(y,p):
    curves=[roc_curve(y[:,j],p[:,j])[:2] for j in range(7)]
    grid=np.unique(np.concatenate([f for f,t in curves]))
    tpr=np.mean([np.interp(grid,f,t) for f,t in curves],axis=0)
    # Preserve the vertical ROC segment at FPR=0 without changing curve area.
    return np.r_[0.,grid],np.r_[0.,tpr]


def save_figure(fig,stem):
    fig.tight_layout()
    fig.savefig(OUT/(stem+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(OUT/(stem+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(stem+'.svg'),bbox_inches='tight')
    plt.close(fig)


def plots(table,ifs,y,predictions,winner):
    fig,ax=plt.subplots(figsize=(12.5,6))
    metrics=['ROC-AUC','AP','Accuracy','Precision','Recall','F1','Specificity','MCC']
    x=np.arange(8); width=.085
    for i,(name,color) in enumerate(zip(NAMES,REFERENCE_PALETTE)):
        values=table.set_index('model').loc[name,metrics].to_numpy(dtype=float)
        ax.bar(x+(i-3)*width,values,width*.9,color=color,alpha=.5,
               edgecolor='#555555',linewidth=.5,label=name)
    ax.set_xticks(x,['ROC-AUC','AP','Accuracy\n(exact match)','Precision','Recall','F1','Specificity','MCC'])
    plt.setp(ax.get_xticklabels(),rotation=0,ha='center',fontsize=9)
    ax.set_ylim(0,1.05)
    ax.set_ylabel('Test score')
    ax.legend(ncol=7,loc='upper center',bbox_to_anchor=(.5,1.16),frameon=False,fontsize=8,
              columnspacing=.8,handlelength=1.2,handletextpad=.4)
    ax.spines[['top','right']].set_visible(False)
    save_figure(fig,'figure_3A_algorithm_metrics')
    fig,ax=plt.subplots(figsize=(12.5,6))
    for name,color in zip(NAMES,REFERENCE_PALETTE):
        f,t=macro_curve(y,predictions[name]); value=table.loc[table.model.eq(name),'ROC-AUC'].iloc[0]
        ax.plot(f,t,color=color,lw=2,label=f'{name} ({value:.3f})')
    ax.plot([0,1],[0,1],'--',color='#777777',lw=1)
    ax.set(xlabel='False-positive rate',ylabel='Mean true-positive rate',title='Macro-average ROC by algorithm')
    ax.set_xlim(-.02,1.02)
    ax.set_ylim(-.02,1.02)
    ax.legend(loc='lower right',frameon=False)
    save_figure(fig,'figure_3B_algorithm_macro_roc')
    fig,ax=plt.subplots(figsize=(12.5,6))
    for idx,(name,color) in enumerate(zip(NAMES[:-1],REFERENCE_PALETTE)):
        sub=ifs.loc[ifs.model.eq(name)].sort_values('n_features')
        best=sub.loc[sub.validation_macro_auc.idxmax()]
        ax.plot(sub.n_features,sub.validation_macro_auc,marker=['o','s','^','D','v','P'][idx],
                linestyle=['-','--','-.',':','-','--'][idx],color=color,
                label=f'{name} [{int(best.n_features)}, {best.validation_macro_auc:.3f}]',lw=2)
        ax.scatter(best.n_features,best.validation_macro_auc,marker='*',s=160,color=color,edgecolors='#444444',zorder=4)
    ax.set(xlabel='Number of features from the mRMR ranking',ylabel='Validation macro ROC-AUC',
           title='mRMR-IFS: stars mark validation maxima')
    ax.set_ylim(.60,1.0)
    ax.legend(loc='lower right',frameon=True,facecolor='white',edgecolor='#cccccc',
              framealpha=.95,fontsize=8,title='Model [features, peak AUC]',title_fontsize=8)
    ax.text(.02,.025,'1D-CNN: raw-sequence input\nmRMR-IFS not applicable',
            transform=ax.transAxes,fontsize=8,va='bottom')
    save_figure(fig,'figure_3C_mrmr_ifs')
    fig,ax=plt.subplots(figsize=(12.5,6))
    class_rows=[]
    for j,color in enumerate(REFERENCE_PALETTE):
        f,t,_=roc_curve(y[:,j],predictions[winner][:,j]); value=roc_auc_score(y[:,j],predictions[winner][:,j])
        enzyme=['Oxidoreductases','Transferases','Hydrolases','Lyases','Isomerases','Ligases','Translocases'][j]
        ax.plot(f,t,color=color,lw=2,label=f'EC{j+1} {enzyme} ({value:.3f})')
        class_rows.append({'ec_class':j+1,'roc_auc':value,'model':winner,'support':int(y[:,j].sum())})
    ax.plot([0,1],[0,1],'--',color='#777777',lw=1)
    ax.set(xlabel='False-positive rate',ylabel='True-positive rate',title=f'Class-specific ROC: {winner}')
    ax.set_xlim(-.02,1.02)
    ax.set_ylim(-.02,1.02)
    ax.legend(frameon=False,loc='lower right')
    save_figure(fig,'figure_3D_per_class_roc')
    pd.DataFrame(class_rows).to_csv(OUT/(f'{winner.replace(" ","_")}_per_class_auc.csv'),index=False)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'models').mkdir(exist_ok=True)
    (OUT/'predictions').mkdir(exist_ok=True)
    log('Starting fixed-seed mRMR-MID/IFS comparison; CPU threads=4')
    master_path=ROOT/'data/processed/archaea_swissprot_ec_master.csv'
    split_path=ROOT/'results/3.2_random_split_performance/split_manifest.csv'
    data=pd.read_csv(master_path)
    splits=pd.read_csv(split_path).set_index('accession').split_random
    labels=[f'label_ec{i}' for i in range(1,8)]
    y=data[labels].to_numpy(dtype=int)
    split=data.accession.map(splits)
    assert split.notna().all()
    assert pd.DataFrame({'group':data.sequence_sha256,'split':split}).groupby('group')['split'].nunique().max()==1
    tr,va,te=[split.eq(s).to_numpy() for s in ['train','validation','test']]
    spec=importlib.util.spec_from_file_location('feature_model',ROOT/'scripts/07_run_lightgbm_auc_model.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    x=mod.load_features(data,ROOT/'data/processed/aac_dpc_features.npy')
    assert x.shape[1]==432 and np.isfinite(x).all()
    config={'seed':SEED,'grid':GRID,'mrmr':'MID; train-only 10-quantile-bin discrete MI; mean seven-label relevance',
        'selection':'Per-model validation maximum; smaller prefix wins exact ties; no test-driven choices',
        'cnn':'raw sequences; 1024 residues; first/last 512 for long sequences; embedding16; conv32 kernels3/5/7; masked max pool',
        'epochs':60,'patience':8,'single_configuration_per_algorithm':True,
        'split_counts':{'train':int(tr.sum()),'validation':int(va.sum()),'test':int(te.sum())},
        'training_accessions_sha256':hashlib.sha256('\n'.join(data.loc[tr,'accession']).encode()).hexdigest(),
        'master_sha256':hashlib.sha256(master_path.read_bytes()).hexdigest(),
        'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),
        'python':platform.python_version(),'torch':torch.__version__,
        'caveat':'Existing test split has been inspected in prior project development; not an untouched external confirmation.'}
    save_json(OUT/'experiment_config.json',config)
    rank_path=OUT/'mrmr_ranking.csv'
    order=pd.read_csv(rank_path).feature_index.to_numpy() if rank_path.exists() else mrmr(x[tr],y[tr])
    log('mRMR ranking ready')
    rows=[]; selections=[]
    for name in NAMES[:-1]:
        slug=name.replace(' ','_')
        record=OUT/'models'/f'{slug}_selection.json'
        if record.exists():
            info=json.loads(record.read_text()); rows.extend(info['ifs']);selections.append(info['selection']);log(f'Resumed completed {name}');continue
        best_score=-np.inf;best_pack=None;best_k=None;local=[]
        for k in GRID:
            began=time.time()
            log(f'Training {name}: top {k}')
            cols=order[:k]
            pack=train_model(name,x[tr][:,cols],y[tr],x[va][:,cols],y[va])
            vp=predict(pack,x[va][:,cols])
            score=roc_auc_score(y[va],vp,average='macro')
            row={'model':name,'n_features':k,'validation_macro_auc':score,'seconds':time.time()-began}
            local.append(row); rows.append(row)
            pd.DataFrame(rows).to_csv(OUT/'ifs_validation_results.csv',index=False)
            log(f'Finished {name} k={k}, validation AUC={score:.5f}, seconds={row["seconds"]:.1f}')
            if score>best_score:
                best_score,best_pack,best_k=score,pack,k
        best_pack['columns']=order[:best_k]
        joblib.dump(best_pack,OUT/'models'/f'{slug}.joblib')
        selected={'model':name,'n_features':best_k,'validation_macro_auc':best_score}
        selections.append(selected)
        save_json(record,{'selection':selected,'ifs':local})
    cnn_path=OUT/'models/1D-CNN_selection.json'
    if cnn_path.exists():
        selections.append(json.loads(cnn_path.read_text())['selection'])
    else:
        seq=encode_sequences(data.sequence)
        log('Training raw-sequence CNN')
        pack=train_model('1D-CNN',seq[tr],y[tr],seq[va],y[va])
        score=roc_auc_score(y[va],predict(pack,seq[va]),average='macro')
        pack['columns']=None
        joblib.dump(pack,OUT/'models/1D-CNN.joblib')
        selected={'model':'1D-CNN','n_features':0,'validation_macro_auc':score}
        selections.append(selected);save_json(cnn_path,{'selection':selected})
    selected_table=pd.DataFrame(selections)
    winner=selected_table.sort_values('validation_macro_auc',ascending=False).iloc[0]['model']
    selected_table.to_csv(OUT/'validation_model_selection.csv',index=False)
    save_json(OUT/'selection_frozen_before_test.json',{'selected_model':winner,'selection':selections})
    log(f'Selection frozen: {winner}; now evaluating held-out test')
    table=[];predictions={}
    for name in NAMES:
        pack=joblib.load(OUT/'models'/f'{name.replace(" ","_")}.joblib')
        inputs=encode_sequences(data.sequence) if name=='1D-CNN' else x[:,pack['columns']]
        vp=predict(pack,inputs[va]);cut=thresholds(y[va],vp);tp=predict(pack,inputs[te])
        predictions[name]=tp
        table.append({'model':name,**metrics(y[te],tp,cut)})
        frame=pd.DataFrame(tp,columns=[f'score_ec{i}' for i in range(1,8)])
        frame.insert(0,'accession',data.loc[te,'accession'].to_numpy())
        for j,label in enumerate(labels):frame[label]=y[te,j]
        frame.to_csv(OUT/'predictions'/f'{name.replace(" ","_")}_test.csv',index=False)
        save_json(OUT/'models'/f'{name.replace(" ","_")}_thresholds.json',cut.tolist())
    table=pd.DataFrame(table)
    table.to_csv(OUT/'algorithm_test_metrics.csv',index=False)
    ifs=pd.DataFrame(rows)
    plots(table,ifs,y[te],predictions,'LightGBM')
    best=table.loc[table.model.eq(winner)].iloc[0]
    save_json(OUT/'run_summary.json',{'status':'completed','winner_by_validation':winner,'selected_test_metrics':best.to_dict(),
        'ifs_grid':GRID,'seed':SEED,'model_count':7,'active_png_count':4})
    log('COMPLETED all seven models, grid IFS, test metrics and four figure pairs')


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
