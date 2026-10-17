"""Stage Figure 3 with SVM class ROC and user template layout; no training."""
import importlib
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

m = importlib.import_module('27_report_five_fold')
ROOT = m.ROOT
OUT = ROOT/'results/svm_migration_20261008/staged_results/3.2_feature_selection_model_comparison'

def plot_ifs(ifs):
    """Emphasize the selected model by drawing order, never by changing scores."""
    m.style()
    plt.rcParams.update({'font.size':22, 'axes.labelsize':24, 'axes.titlesize':26,
                         'xtick.labelsize':21, 'ytick.labelsize':21})
    fig,ax = plt.subplots(figsize=(16,8.5))
    for i,name in enumerate(m.NAMES[:-1]):
        sub=ifs[ifs.model.eq(name)].sort_values('n_features')
        xs=sub.n_features.to_numpy(); y=sub.validation_macro_auc_mean.to_numpy(); sd=sub.validation_macro_auc_sd.to_numpy()
        best=int(np.argmax(y)); col=m.COLORS[i]; is_svm=name=='RBF-SVM'
        ax.plot(xs,y,color=col,marker=['o','s','^','D','v','P'][i],ls=['-','--','-.',':'][i%4],
                linewidth=3.2 if is_svm else 1.8, zorder=10 if is_svm else 3,
                label=f'{name} [{xs[best]}, {y[best]:.3f} ± {sd[best]:.3f}]')
        ax.fill_between(xs,y-sd,y+sd,color=col,alpha=.10,zorder=1)
        ax.scatter(xs[best],y[best],s=230 if is_svm else 170,marker='*',color=col,
                   edgecolors='#555555',zorder=11 if is_svm else 5)
    ax.set(ylim=(.6,1),xlabel='Number of features from the mRMR ranking',
           ylabel='Validation macro ROC-AUC',title='mRMR-IFS: stars mark validation maxima')
    legend=ax.legend(loc='lower right',fontsize=17,title='Model [features, peak AUC ± SD]',title_fontsize=18)
    for label in legend.get_texts():
        if label.get_text().startswith('RBF-SVM '): label.set_fontweight('bold')
    # Keep the legend above the explicitly raised SVM artists.
    legend.set_zorder(20)
    m.save(fig,OUT,'figure_3C_mrmr_ifs')

def compose_figure3():
    composer=m.module('17_export_composite_figures.py','svm_figure3_composer')
    composer.ARCHIVE=ROOT/'results/svm_migration_20261008/staging_previous'
    composer.compose(OUT,3,[[('figure_3A_algorithm_metrics',1.5),('figure_3B_algorithm_macro_roc',1)],
                           [('figure_3C_mrmr_ifs',1.5),('figure_3D_per_class_roc',1)]])

def main(ifs_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT/'results/five_fold_cv_20261007/staged_results/3.2_feature_selection_model_comparison'
    metrics = pd.read_csv(source/'algorithm_test_metrics_numeric.csv').set_index('model')
    ifs = pd.read_csv(source/'ifs_validation_results_numeric.csv')
    if ifs_only:
        plot_ifs(ifs)
        compose_figure3()
        print('STAGED Figure 3C styling only; data and panels A/B/D unchanged')
        return
    frames = {}
    for name in m.NAMES:
        loaded, _ = m.load_five('selected__random__', '__'+name.replace(' ', '_'))
        frames[name] = [p for p,r in loaded]
    m.style()
    plt.rcParams.update({'font.size':22, 'axes.labelsize':24, 'axes.titlesize':26,
                         'xtick.labelsize':21, 'ytick.labelsize':21})
    fig, ax = plt.subplots(figsize=(16,8.5))
    x = np.arange(8); w = .085
    for i,name in enumerate(m.NAMES):
        y = [metrics.loc[name,k+'_mean'] for k in m.METRICS]
        sd = [metrics.loc[name,k+'_sd'] for k in m.METRICS]
        ax.bar(x+(i-3)*w,y,w*.9,yerr=sd,color=m.COLORS[i],alpha=.5,capsize=2,
               edgecolor='#777777',lw=.5,label=name)
    ax.set_xticks(x,m.METRICS); ax.set(ylim=(0,1.06),ylabel='Test score (mean ± SD)')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.17),ncol=7,fontsize=16,
              frameon=False,handlelength=.7,handletextpad=.3,columnspacing=.8)
    ax.spines[['top','right']].set_visible(False)
    m.save(fig,OUT,'figure_3A_algorithm_metrics')

    fig,ax = plt.subplots(figsize=(10.5,8.5))
    for i,name in enumerate(m.NAMES): m.roc(ax,frames[name],name,m.COLORS[i],ls=['-','--','-.',':'][i%4])
    m.roc_axes(ax);ax.set_title('Macro-average ROC by algorithm')
    ax.legend(loc='lower right',fontsize=16,frameon=False)
    m.save(fig,OUT,'figure_3B_algorithm_macro_roc')

    plot_ifs(ifs)

    fig,ax=plt.subplots(figsize=(10.5,8.5))
    for c in range(7):m.roc(ax,frames['RBF-SVM'],f'EC{c+1} {m.CLASSES[c]}',m.COLORS[c],c)
    m.roc_axes(ax);ax.set_title('Class-specific ROC: RBF-SVM')
    ax.legend(loc='lower right',fontsize=15,frameon=False)
    m.save(fig,OUT,'figure_3D_per_class_roc')
    compose_figure3()
    template=ROOT/'手稿/结果图/图3.png'
    record={'status':'preview_only; not published', 'final_model':'RBF-SVM',
        'source_run':'results/five_fold_cv_20261007', 'new_training':False,
        'outer_folds':5, 'SD':'sample standard deviation, ddof=1',
        'template':str(template.relative_to(ROOT)), 'template_sha256':hashlib.sha256(template.read_bytes()).hexdigest(),
        'panels_ABC':'all seven original algorithms retained as comparator evidence; six IFS feature algorithms',
        'panel_D':'existing five-fold RBF-SVM predictions; no LightGBM predictions',
        'score_type':'sigmoid decision-function scores, not calibrated probabilities',
        'accuracy':'multilabel exact-match accuracy',
        'layout':'template left column 1.5 times right column, A/B top and C/D bottom'}
    (OUT/'svm_figure3_provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print('STAGED SVM Figure 3 preview; formal results and templates unchanged')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--ifs-only',action='store_true',help='Redraw C and compose; no training or changes to A/B/D')
    main(ifs_only=parser.parse_args().ifs_only)
