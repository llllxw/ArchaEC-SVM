"""Recompute all sweep AUCs and audit vector exports from saved evidence."""
from pathlib import Path
import json,re,hashlib
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/3.3_generalization_ladder'
RUN=OUT/'expansion_20260924'
labels=[f'label_ec{i}' for i in range(1,8)]
scores=[f'prob_ec{i}' for i in range(1,8)]
data=pd.read_csv(ROOT/'data/processed/archaea_swissprot_ec_master.csv')
rows=[]
for threshold in range(90,0,-10):
    folder=OUT/f'homology_id{threshold}'
    if not (folder/'optimized_lightgbm/optimized_auc_metrics.csv').exists(): continue
    splits=pd.read_csv(folder/'split_manifest.csv')
    assert len(splits)==len(data) and splits.accession.is_unique
    assert splits.groupby('homology_cluster_id').split_homology.nunique().max()==1
    joined=data[['accession','sequence_sha256']+labels].merge(splits,on='accession',validate='one_to_one')
    exact_overlap=int((joined.groupby('sequence_sha256').split_homology.nunique()>1).sum())
    assert exact_overlap==0, f'{threshold}: identical sequences cross partitions'
    pred=pd.read_csv(folder/'optimized_lightgbm/optimized_test_probabilities.csv')
    assert set(pred.accession)==set(splits.loc[splits.split_homology.eq('test'),'accession'])
    pred=pred.merge(data[['accession']+labels],on='accession',validate='one_to_one')
    truth=pred[labels].to_numpy(); probability=pred[scores].to_numpy()
    assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
    macro=roc_auc_score(truth,probability,average='macro')
    micro=roc_auc_score(truth,probability,average='micro')
    recorded=pd.read_csv(folder/'optimized_lightgbm/optimized_auc_metrics.csv').iloc[0]
    assert abs(macro-recorded.macro_roc_auc)<1e-12
    assert abs(micro-recorded.micro_roc_auc)<1e-12
    pc=pd.read_csv(folder/'optimized_lightgbm/optimized_per_class_auc.csv').sort_values('ec_class')
    recomputed=[roc_auc_score(truth[:,j],probability[:,j]) for j in range(7)]
    assert np.allclose(recomputed,pc.roc_auc,atol=1e-12,rtol=0)
    rows.append({'identity_percent':threshold,'macro_auc':macro,'micro_auc':micro,
        'training_seed':int(recorded.seed),
        'minimum_class_auc':min(recomputed),'n_test':len(pred),
        'n_clusters':splits.homology_cluster_id.nunique(),'test_clusters':splits.loc[splits.split_homology.eq('test'),'homology_cluster_id'].nunique(),
        'cluster_overlap':0,'exact_sequence_overlap':exact_overlap,
        'source':'retained' if threshold in [50,40] else 'new_run',
        'prediction_sha256':hashlib.sha256((folder/'optimized_lightgbm/optimized_test_probabilities.csv').read_bytes()).hexdigest()})
table=pd.DataFrame(rows)
table.to_csv(RUN/'verified_threshold_metrics.csv',index=False)

# Recover each complete clustering manifest from its command log; script 09's
# root manifest is overwritten by each new threshold but command logs retain it.
manifests=[]
for log in RUN.glob('id*_cluster.log'):
    raw=log.read_bytes()
    starts=list(re.finditer(rb'\{\r?\n  "started_at_utc"',raw))
    if starts:
        payload=raw[starts[-1].start():]
        try: content=payload.decode('utf-8')
        except UnicodeDecodeError: content=payload.decode('gb18030')
        manifest=json.loads(content)
        manifests.append(manifest)
        (RUN/(log.stem+'_manifest.json')).write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
(RUN/'all_clustering_manifests.json').write_text(json.dumps(manifests,indent=2,ensure_ascii=False),encoding='utf-8')

svgs=[]
for directory in ['3.1_dataset_characterization','3.2_feature_selection_model_comparison','3.3_generalization_ladder','3.5_seven_class_interpretability']:
    for path in (ROOT/'results'/directory).glob('*.svg'):
        svg=ET.parse(path).getroot()
        assert not list(svg.iter('{http://www.w3.org/2000/svg}image'))
        ids=[el.attrib['id'] for el in svg.iter() if 'id' in el.attrib]
        assert len(ids)==len(set(ids)), f'Duplicate SVG IDs: {path}'
        svgs.append(str(path.relative_to(ROOT)))
result={'status':'verified' if len(rows)==9 else 'partial','thresholds_verified':len(rows),
        'svg_count':len(svgs),'svg_vector_only':True,'unique_svg_ids':True,
        'all_macro_auc_ge_0_8':bool((table.macro_auc>=.8).all()),
        'all_class_auc_ge_0_8':bool((table.minimum_class_auc>=.8).all()),
        'verified_files':svgs}
comparison=ROOT/'results/3.2_feature_selection_model_comparison'
pred=pd.read_csv(comparison/'predictions/LightGBM_test.csv')
auc=roc_auc_score(pred[labels],pred[[f'score_ec{i}' for i in range(1,8)]],average='macro')
metrics=pd.read_csv(comparison/'algorithm_test_metrics.csv').set_index('model')
assert abs(auc-metrics.loc['LightGBM','ROC-AUC'])<1e-12
perclass=pd.read_csv(comparison/'LightGBM_per_class_auc.csv')
assert perclass.model.eq('LightGBM').all() and abs(perclass.roc_auc.mean()-auc)<1e-12
roc_svg=(OUT/'figure_4D_homology_roc_curves.svg').read_text(encoding='utf-8')
assert roc_svg.count('Class ROC-AUC')==9
result['figure3D_lightgbm_macro_auc']=float(auc)
result['figure4D_legend_count']=9
a=pd.read_csv(OUT/'homology_id20/split_manifest.csv')
b=pd.read_csv(OUT/'homology_id10/split_manifest.csv')
cross=a.merge(b,on='accession',suffixes=('_20','_10'),validate='one_to_one')
equivalent=bool((cross.groupby('homology_cluster_id_20').homology_cluster_id_10.nunique()==1).all()
    and (cross.groupby('homology_cluster_id_10').homology_cluster_id_20.nunique()==1).all())
result['threshold20_10_identical_cluster_partitions']=equivalent
result['threshold20_10_test_overlap']=len(set(a.loc[a.split_homology.eq('test'),'accession'])&set(b.loc[b.split_homology.eq('test'),'accession']))
(RUN/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(table[['identity_percent','macro_auc','micro_auc','minimum_class_auc','n_test','n_clusters']].to_string(index=False))
print(json.dumps({k:v for k,v in result.items() if k!='verified_files'},indent=2))
progress=json.loads((RUN/'progress.json').read_text(encoding='utf-8'))
report=['# LightGBM 同源阈值扩展实验结果','',
    '## Material Passport','',
    f"运行状态：{progress['status']}；验证状态：{result['status']}。日期：2026-09-24。",
    '模型：前期固定配置 LightGBM；输入：432 维 AAC+DPC+理化描述符。',
    '新运行：90%、80%、70%、60%、30%、20%、10%；保留：50%、40%。',
    f"记录的运行耗时：{progress.get('elapsed_seconds',0):.1f} 秒。未上传数据；未根据测试分数调参。",'',
    '## 已验证结果','',
    '| 阈值 | 宏平均 AUC | 微平均 AUC | 最低类别 AUC | 测试蛋白数 | 同源簇总数 | 训练种子 |',
    '|---|---:|---:|---:|---:|---:|---:|']
for row in table.itertuples():
    report.append(f'| {row.identity_percent}% | {row.macro_auc:.4f} | {row.micro_auc:.4f} | {row.minimum_class_auc:.4f} | {row.n_test} | {row.n_clusters} | {row.training_seed} |')
report += ['', '## 核查与边界','',
    '逐阈值核查登录号覆盖、簇隔离、完全相同序列隔离、预测范围、逐类 AUC、宏平均和微平均 AUC。重新计算与记录值在 1e-12 容差内一致。',
    '不同阈值使用不同测试蛋白集合，不是同一测试集的严格嵌套难度比较。10%/20% 指 MMseqs2 聚类阈值，并非已证明所有跨分区蛋白的真实一致性低于该值。',
    '新增七档固定训练种子为 42；保留的 50%/40% 旧结果分别使用 92/82。实测核对 EC1 保存模型，旧 50% 与新 90% 的 get_params() 差异仅为 random_state；其他超参数一致。该曲线保留历史种子差异，不是统一种子下的重复实验。',
    'AUC 点估计未附重复划分或多种子置信区间。未达到 0.80 的结果完整保留，不能据此声称所有隔离条件或所有类别 AUC 均超过 0.80。',
    f"20% 与 10% 的簇成员分组是否完全一致：{equivalent}；两档均得到 1,116 个簇。两档划分种子不同，测试集各有 1,236 条蛋白，其中 {result['threshold20_10_test_overlap']} 条重叠。因此 0.6547 与 0.6681 的差异不能解释为降低阈值改善了远缘泛化；这两点没有形成新的簇隔离层级。",
    '图 4D 现展示九个阈值的 3×3 ROC 面板，图 4E 展示九个阈值与七类酶的 AUC 热图；总体曲线见图 4C。',
    '图 3D 使用本轮模型比较的 LightGBM（宏平均 AUC=0.968374）；其参数与历史泛化 LightGBM 不同，不能将它们称为同一个已训练模型。',
    '', '## 产物','',
    '完整核查表：`verified_threshold_metrics.csv`。机器可读核查：`verification.json`。运行命令、进程、状态：`progress.json`。各阈值的聚类、划分、训练日志保存在本目录。',
    '各阈值模型与逐样本预测保存在上一级 `homology_idXX/optimized_lightgbm/`。聚类命令和输入哈希记录于 `all_clustering_manifests.json`。',
    '统一重绘：`scripts/17_export_composite_figures.py`；只读复算与结果记录：`scripts/19_verify_homology_and_figures.py`。']
(RUN/'experiment_result.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
