"""Verify saved scores and render feature-only Figure 5 in reference layout."""
import importlib.util,json,math,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score
from plot_style import apply_arial_style

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/3.4_multifunction_interpretability'
RUN=OUT/'feature_ablation_complete_20260924'
ARCHIVE=ROOT/'results/_archive_figures_20260925_final_style/3.4_multifunction_interpretability'
# User-approved grouped hues: neutral full, cool singles, warm pairs.
COLORS={'full':'#555555','aac':'#5B6F82','dpc':'#6D91B3','phys':'#819DA5',
        'aac_dpc':'#C38B58','aac_phys':'#AB263A','dpc_phys':'#CE7D77'}

def plot_panel(table,key,stem,ylabel):
    apply_arial_style()
    fig,ax=plt.subplots(figsize=(14,5.2))
    x=np.arange(7);values=table[key].to_numpy()
    low=math.floor((values.min()-.025)*20)/20
    high=min(1.0,math.ceil((values.max()+.06)*100)/100)
    span=high-low
    for start,end,shade,label in [(-.6,.5,'#F2F2F2','Full representation'),
                                (.5,3.5,'#E8EFF6','Single feature group'),
                                (3.5,6.6,'#F5E9DF','Paired feature groups')]:
        ax.axvspan(start,end,facecolor=shade,alpha=.5,zorder=0)
        ax.text((start+end)/2,high-.04*span,label,ha='center',va='top',fontsize=15)
    for i,(value,color) in enumerate(zip(values,[COLORS[tag] for tag in table.tag])):
        ax.vlines(i,low,value,color=color,alpha=.5,lw=1.8,zorder=2)
        ax.scatter(i,value,s=150,facecolors='none',edgecolors=color,linewidths=1.8,alpha=.5,zorder=3)
        ax.scatter(i,value,s=42,c=color,alpha=.5,zorder=4)
        ax.text(i,value+.025*span,f'{value:.4f}',ha='center',va='bottom',fontsize=16)
    labels=['Full\n(432)','AAC\n(20)','DPC\n(400)','Physicochemical\n(12)',
            'AAC + DPC\n(420)','AAC +\nPhysicochemical\n(32)','DPC +\nPhysicochemical\n(412)']
    ax.set_xticks(x,labels,fontsize=15)
    ax.set_xlim(-.6,6.6);ax.set_ylim(low,high)
    ax.set_ylabel(ylabel,fontsize=20)
    ax.tick_params(axis='y',labelsize=16)
    ax.spines[['top','right']].set_visible(True)
    ax.grid(False)
    fig.tight_layout()
    for extension in ['svg','pdf','png']:
        path=OUT/f'{stem}.{extension}'
        if path.exists() and not (ARCHIVE/path.name).exists():shutil.copy2(path,ARCHIVE/path.name)
        fig.savefig(path,dpi=300,bbox_inches='tight')
    plt.close(fig)

def main():
    status=json.loads((RUN/'run_status.json').read_text())
    assert status['status']=='completed','Training incomplete'
    table=pd.read_csv(OUT/'feature_ablation_auc_ap.csv')
    assert len(table)==7
    labels=[f'label_ec{i}' for i in range(1,8)]
    scores=[f'score_ec{i}' for i in range(1,8)]
    reference=None
    for row in table.itertuples():
        p=pd.read_csv(RUN/'predictions'/f'{row.tag}_test.csv')
        if reference is None:reference=p[['accession']+labels]
        pd.testing.assert_frame_equal(reference,p[['accession']+labels])
        assert abs(roc_auc_score(p[labels],p[scores],average='macro')-row.macro_roc_auc)<1e-12
        assert abs(average_precision_score(p[labels],p[scores],average='macro')-row.macro_ap)<1e-12
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    old=['figure_5A_multifunction_auc','figure_5B_feature_ablation_auc','figure_5C_annotation_stratified_auc']
    for stem in old:
        for extension in ['png','pdf','svg']:
            path=OUT/f'{stem}.{extension}'
            if path.exists():
                assert path.resolve().parent==OUT.resolve()
                dest=ARCHIVE/path.name
                if not dest.exists():shutil.copy2(path,dest)
                path.unlink()  # recoverable exact-file removal from active figure set
    order=['full','aac','dpc','phys','aac_dpc','aac_phys','dpc_phys']
    table=table.set_index('tag').loc[order].reset_index()
    plot_panel(table,'macro_roc_auc','figure_5A_feature_ablation_auc','Macro ROC-AUC')
    plot_panel(table,'macro_ap','figure_5B_feature_ablation_ap','Macro average precision (AP)')
    spec=importlib.util.spec_from_file_location('exporter',ROOT/'scripts/17_export_composite_figures.py')
    exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    exporter.ARCHIVE=ARCHIVE.parent
    exporter.compose(OUT,5,[[('figure_5A_feature_ablation_auc',1)],[('figure_5B_feature_ablation_ap',1)]])
    verification={'status':'verified','variants':7,'n_test':len(reference),'auc_ap_recomputed':True,
                  'old_auc_max_abs_difference':float(table.previous_auc_difference.abs().max()),
                  'new_variants':['AAC + Physicochemical','DPC + Physicochemical'],
                  'alpha':.5,'font':'Arial','full_model_reused':True,
                  'feature_colors':COLORS,
                  'archived_stems':old,'seed_policy':status['seed_policy']}
    (RUN/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    report=['# 完整特征组消融：AUC 与 AP','', '## Material Passport','',
            f"状态：completed / verified；运行时间：{status['elapsed_seconds']:.1f} 秒；七类多标签任务；固定随机划分。",'',
            '完整模型沿用旧 LightGBM；四个旧变体按原配置、原种子复跑以恢复 AP；新增两个双特征组。每组种子与维度见结果 CSV。',
            '所有组的模型超参数一致，但保留了旧实验不同的种子；这是单轮描述性比较，不据小幅差异宣称显著增益。',
            'AP 使用七个 average_precision 的算术平均，不是梯形积分 PR-AUC。完整模型为 432 维，AAC+理化为 32 维，DPC+理化为 412 维。','',
            '| 特征组 | 维度 | 宏平均 AUC | 宏平均 AP |','|---|---:|---:|---:|']
    for row in table.itertuples():report.append(f'| {row.feature_set} | {row.n_features} | {row.macro_roc_auc:.4f} | {row.macro_ap:.4f} |')
    report+=['','图 5 仅保留特征消融内容，A 为 AUC，B 为 AP；圆点和细杆 alpha=0.5。背景分区区分完整、单组和双组输入，不沿用模板中不存在的网络组件名称。',
             '纵轴为截取区间，用于显示数值差异；必须结合坐标和点上数值阅读，不将杆长比例视为效应大小。未绘制缺乏重复实验依据的误差线。',
             f"原 AUC 最大绝对复算差异：{verification['old_auc_max_abs_difference']:.12g}。",
             '原图 5A/5C 和旧柱状图移至 `_archive_figures_20260924_before_full_ablation/3.4_multifunction_interpretability/`，辅助任务及注释分析的数据文件保留。']
    (RUN/'experiment_result.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(table[['feature_set','n_features','macro_roc_auc','macro_ap']].to_string(index=False))
    print(json.dumps(verification,indent=2))

if __name__=='__main__':main()
