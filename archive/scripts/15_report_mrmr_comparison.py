"""Validate saved test scores and write reproducible Chinese results for 3.2."""
from pathlib import Path
import hashlib
import json
import importlib.metadata
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import font_manager
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc as integrate_auc

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/3.2_feature_selection_model_comparison'


def main():
    summary=json.loads((OUT/'run_summary.json').read_text(encoding='utf-8'))
    config=json.loads((OUT/'experiment_config.json').read_text(encoding='utf-8'))
    table=pd.read_csv(OUT/'algorithm_test_metrics.csv')
    validation=pd.read_csv(OUT/'validation_model_selection.csv')
    ifs=pd.read_csv(OUT/'ifs_validation_results.csv')
    ranking=pd.read_csv(OUT/'mrmr_ranking.csv')
    assert len(ranking)==432 and ranking.feature_index.nunique()==432
    assert len(ifs)==36 and len(table)==7
    winner=validation.loc[validation.validation_macro_auc.idxmax(),'model']
    assert winner==summary['winner_by_validation']
    manifest=pd.read_csv(ROOT/'results/3.2_random_split_performance/split_manifest.csv')
    expected=set(manifest.loc[manifest.split_random.eq('test'),'accession'])
    master=pd.read_csv(ROOT/'data/processed/archaea_swissprot_ec_master.csv').set_index('accession')
    train_set=set(manifest.loc[manifest.split_random.eq('train'),'accession'])
    training_order=[a for a in master.index if a in train_set]
    assert hashlib.sha256('\n'.join(training_order).encode()).hexdigest()==config['training_accessions_sha256']
    checks=[]
    for row in table.itertuples(index=False):
        name=row.model
        frame=pd.read_csv(OUT/'predictions'/f'{name.replace(" ","_")}_test.csv')
        assert len(frame)==len(expected) and frame.accession.is_unique and set(frame.accession)==expected
        y=frame[[f'label_ec{i}' for i in range(1,8)]].to_numpy()
        assert np.array_equal(y,master.loc[frame.accession,[f'label_ec{i}' for i in range(1,8)]].to_numpy())
        p=frame[[f'score_ec{i}' for i in range(1,8)]].to_numpy()
        assert np.isfinite(p).all()
        auc=roc_auc_score(y,p,average='macro')
        ap=average_precision_score(y,p,average='macro')
        reported=table.loc[table.model.eq(name)].iloc[0]
        assert np.isclose(auc,reported['ROC-AUC'],atol=1e-12)
        assert np.isclose(ap,reported.AP,atol=1e-12)
        curves=[roc_curve(y[:,j],p[:,j])[:2] for j in range(7)]
        grid=np.unique(np.concatenate([f for f,t in curves]))
        curve_auc=integrate_auc(grid,np.mean([np.interp(grid,f,t) for f,t in curves],axis=0))
        checks.append({'model':name,'n_test':len(frame),'macro_auc':auc,'macro_curve_area':curve_auc,
                       'integration_difference':abs(auc-curve_auc),'labels_match_master':True})
        best=validation.loc[validation.model.eq(name)].iloc[0]
        if name!='1D-CNN':
            assert np.isclose(best.validation_macro_auc,ifs.loc[ifs.model.eq(name),'validation_macro_auc'].max())
            ranking.iloc[:int(best.n_features)].to_csv(OUT/'models'/f'{name.replace(" ","_")}_selected_features.csv',index=False)
    figure_files=sorted(OUT.glob('figure_3*.png'))
    assert len(figure_files)==4
    hashes={}
    for path in figure_files:
        with Image.open(path) as img:
            img.verify()
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        assert path.with_suffix('.pdf').exists()
    packages={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','lightgbm','xgboost','torch','matplotlib']}
    source_hash=hashlib.sha256((ROOT/'scripts/14_mrmr_model_comparison.py').read_bytes()).hexdigest()
    arial_path=font_manager.findfont('Arial',fallback_to_default=False)
    (OUT/'verification.json').write_text(json.dumps({'status':'verified','checks':checks,'figure_sha256':hashes,
        'arial_font_path':arial_path,
        'packages':packages,'training_script_sha256':source_hash,'training_accessions_verified':True},indent=2),encoding='utf-8')
    best=table.loc[table.model.eq(winner)].iloc[0]
    chosen=validation.loc[validation.model.eq(winner)].iloc[0]
    perclass=pd.read_csv(OUT/'selected_model_per_class_auc.csv')
    comparison='；'.join(f'{row.model}为{table.loc[table.model.eq(row.model),"ROC-AUC"].iloc[0]:.3f}' for row in table.itertuples())
    selected_text='原始序列输入' if winner=='1D-CNN' else f'mRMR排序前{int(chosen.n_features)}个特征'
    feature_text='；'.join(f'{r.model}选择{int(r.n_features)}维' for r in validation.itertuples() if r.model!='1D-CNN')
    text=f'''## 3.2 mRMR特征筛选与机器学习、深度学习模型比较

以20维氨基酸组成、400维二肽组成及12维理化性质构成432维初始描述符，在训练集内采用最小冗余最大相关（mRMR）的互信息差值准则生成特征排序。相关性定义为特征与七个一级EC二元标签互信息的等权平均，冗余为与已选特征互信息的平均；连续特征使用训练集分位点离散化。沿该排序的前16、32、64、128、256和432维开展网格式增量特征选择（IFS），以验证集宏平均ROC-AUC确定各描述符模型的特征数量。{feature_text}（图3C）。该结果限于预设网格中的最优节点，不代表对所有特征子集完成穷举。

在相同随机蛋白留出划分下比较逻辑回归、RBF-SVM、随机森林、XGBoost、LightGBM、DNN和1D-CNN。前六种方法使用描述符，CNN使用原始氨基酸序列。训练、验证和测试集分别包含{config['split_counts']['train']:,}、{config['split_counts']['validation']:,}和{config['split_counts']['test']:,}条蛋白。以验证集宏平均ROC-AUC作为唯一主模型选择依据，选定{winner}，其输入为{selected_text}，验证AUC为{chosen.validation_macro_auc:.3f}。测试集各算法的宏平均ROC-AUC分别为：{comparison}（图3A、B）。

验证集选定的{winner}在测试集上的宏平均ROC-AUC为{best['ROC-AUC']:.3f}，宏平均AP为{best.AP:.3f}，多标签完全匹配率为{best.Accuracy:.3f}，宏平均Precision、Recall、F1、Specificity和MCC分别为{best.Precision:.3f}、{best.Recall:.3f}、{best.F1:.3f}、{best.Specificity:.3f}和{best.MCC:.3f}（图3A）。其中完全匹配率要求一条蛋白的全部标签均预测正确；阈值相关指标的类别阈值只由验证集确定。七个一级EC类别的ROC-AUC介于{perclass.roc_auc.min():.3f}与{perclass.roc_auc.max():.3f}之间（图3D）。

本轮采用固定随机种子和每算法一套预设结构配置，比较包括特征数量选择及验证集早停，尚不构成多随机种子稳健性检验或全面超参数优化。CNN和描述符模型的输入表示不同，因此两者差异属于完整预测流程的比较。该测试划分在前期开发中已有结果被查看，不能视为完全未接触的外部确认集。
'''
    (OUT/'results_3.2_mrmr_draft.md').write_text(text,encoding='utf-8')
    readme='''# Results 3.2 — mRMR-MID, grid IFS and seven-model comparison

Run `python -u scripts/14_mrmr_model_comparison.py` from project root. Validate with
`python scripts/15_report_mrmr_comparison.py`. Fixed seed: 42. All images: Arial,
user reference palette, 300 dpi PNG and vector PDF.

- `figure_3A_algorithm_metrics`: eight metrics in a 2×4 bar-chart figure.
- `figure_3B_algorithm_macro_roc`: seven algorithm macro ROC curves.
- `figure_3C_mrmr_ifs`: measured validation AUC at six mRMR prefix sizes.
- `figure_3D_per_class_roc`: seven EC ROC curves of the validation-selected model.

`mrmr_ranking.csv` records all 432 ranked features and per-label mutual information.
`mrmr_discretization.json` records train-only quantile edges. The implementation
uses MID (mean label relevance minus mean selected-feature redundancy), not plain
MI ranking or a correlation proxy. Prefix sizes are 16,32,64,128,256,432. Lines
between measured IFS points are guides, not additional measured experiments.

`validation_model_selection.csv` determines the winning model, independently of
`algorithm_test_metrics.csv`. Test truth and scores are in `predictions/`; models,
scalers, selected columns and thresholds are in `models/`. `verification.json`
checks test identities, true labels, AUC/AP recomputation and figure hashes.

SVM sigmoid-transformed margins are ranking scores, not calibrated probabilities.
Accuracy means exact label-set match. AP is sklearn average precision, not a
trapezoidal PR integral. No force-nonempty label rule is applied. All threshold
search uses validation data. No uncertainty bars are invented for a single run.

The original Figure 3C and its original LightGBM scores remain in the historical
`3.2_random_split_performance` directory. New figures and selected models are not
equated with the prior 432-feature results. The existing test split was inspected
in earlier project development and is not a new external validation cohort.

## Source

Peng et al. (2005), mRMR: https://doi.org/10.1109/TPAMI.2005.159
'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    print(json.dumps({'winner':winner,'verified_models':len(checks),'verified_figures':len(figure_files),'max_macro_curve_difference':max(c['integration_difference'] for c in checks)},indent=2))


if __name__=='__main__':
    main()
