"""New, non-replacing preview: ordered pairwise AUC means and sample SDs."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from plot_style import apply_arial_style, MODEL_COLOR, BASELINE_COLOR

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/3.5_seven_class_interpretability'
OUT=ROOT/'手稿/结果图/图6B_排序点图预览_20261008'
STEM='figure_6B_sorted_dotplot_preview'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    # Refuse to overwrite even previous previews.
    OUT.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((ROOT/'results/svm_migration_20261008/published_manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    table=pd.read_csv(SOURCE/'pairwise_class_auc_numeric.csv',float_precision='round_trip')
    folds=pd.read_csv(SOURCE/'pairwise_class_auc_folds.csv',float_precision='round_trip')
    assert len(table)==21 and len(folds)==105
    for row in table.itertuples():
        values=folds.loc[folds.class_a.eq(row.class_a)&folds.class_b.eq(row.class_b),'pairwise_roc_auc'].to_numpy()
        assert len(values)==5 and row.pairwise_roc_auc_n_folds==5
        assert np.isclose(values.mean(),row.pairwise_roc_auc_mean,rtol=0,atol=1e-12)
        assert np.isclose(values.std(ddof=1),row.pairwise_roc_auc_sd,rtol=0,atol=1e-12)
    table=table.sort_values(['pairwise_roc_auc_mean','class_a','class_b']).reset_index(drop=True)
    mean=table.pairwise_roc_auc_mean.to_numpy();sd=table.pairwise_roc_auc_sd.to_numpy()
    assert (mean-sd>=.9).all() and (mean+sd<=1).all(), 'Do not clip SD whiskers'
    apply_arial_style()
    plt.rcParams.update({'font.size':16,'axes.labelsize':18,'xtick.labelsize':15,
                         'ytick.labelsize':16,'svg.hashsalt':'pairwise-dotplot-preview-v1'})
    fig,ax=plt.subplots(figsize=(12,12.5))
    fig.subplots_adjust(left=.16,right=.75,top=.87,bottom=.12)
    y=np.arange(len(table))
    for i in range(len(table)):
        if i%2==0:ax.axhspan(i-.5,i+.5,color=BASELINE_COLOR,alpha=.08,zorder=0)
    artists=ax.errorbar(mean,y,xerr=sd,fmt='o',color=MODEL_COLOR,ecolor=MODEL_COLOR,
        markersize=7.5,markeredgecolor='white',markeredgewidth=.8,
        elinewidth=1.8,capsize=4,capthick=1.5,zorder=3)
    assert np.array_equal(artists.lines[0].get_xdata(),mean)
    ax.set_yticks(y,[f'EC{int(a)}–EC{int(b)}' for a,b in zip(table.class_a,table.class_b)])
    ax.set(xlim=(.90,1.0),ylim=(len(table)-.5,-.6),xlabel='Pairwise ROC-AUC')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(.02))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.grid(axis='x',color='#D6D6D6',linewidth=.7,linestyle='--',alpha=.8)
    ax.set_axisbelow(True)
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y',length=0,pad=12)
    ax.text(-.025,1.025,'Enzyme pair',transform=ax.transAxes,ha='right',va='bottom',fontsize=16,fontweight='bold')
    ax.text(1.07,1.025,'Mean ± SD',transform=ax.transAxes,ha='left',va='bottom',fontsize=16,fontweight='bold')
    for i,(mu,sigma) in enumerate(zip(mean,sd)):
        ax.text(1.07,i,f'{mu:.3f} ± {sigma:.3f}',transform=ax.get_yaxis_transform(),
                ha='left',va='center',fontsize=16,color='#333333',clip_on=False)
    fig.text(.16,.96,'Pairwise discrimination of seven enzyme classes',fontsize=23,fontweight='bold',ha='left')
    fig.text(.16,.927,'RBF-SVM · Random five-fold outer tests · Ordered by mean AUC',fontsize=15,color='#555555',ha='left')
    fig.text(.16,.057,'Dots: five-fold means; whiskers: ±1 sample SD (not confidence intervals).',fontsize=12.5,color='#555555')
    fig.text(.16,.034,'Complete single-label proteins; descriptive ranking, not a significance test.',fontsize=12.5,color='#555555')
    for ext in ['svg','pdf','png']:fig.savefig(OUT/f'{STEM}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.2)
    plt.close(fig)
    tree=ET.parse(OUT/f'{STEM}.svg')
    assert not list(tree.iter('{http://www.w3.org/2000/svg}image'))
    for row in manifest['files']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    record={'status':'preview only; not substituted into Figure 6','font':'Arial',
        'chart':'ascending horizontal dot plot, lowest mean at top','xlim':[.9,1],
        'error_bars':'sample SD, ddof=1; not confidence intervals; no clipping',
        'source_sha256':{n:sha(SOURCE/n) for n in ['pairwise_class_auc_numeric.csv','pairwise_class_auc_folds.csv']},
        'all_21_means_and_sds_recomputed':True,'unchanged_published_files':len(manifest['files']),
        'point_color':MODEL_COLOR,'training_performed':False,
        'outputs':{f'{STEM}.{ext}':sha(OUT/f'{STEM}.{ext}') for ext in ['svg','pdf','png']}}
    (OUT/'preview_provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Created new dot-plot preview; verified 21 means/SDs and 212 unchanged published files.',flush=True)
    print(OUT,flush=True)

if __name__=='__main__':main()
