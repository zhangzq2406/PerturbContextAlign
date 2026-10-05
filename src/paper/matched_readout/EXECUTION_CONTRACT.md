# R1执行合同补充（计算前固定）

来源为当前用户批准消息和R0已接受合同。仅执行sci-Plex剂量分支；本轮不再进行总体审计。

## 计算范围

- 六编码器×四视图×三细胞系，72条基础记录；六编码器×四个预定对比×三细胞系，72条配对变化记录。
- 共同2250条件；每细胞系750条件、188药、四剂量。保留原2256条件训练得到的OOF预测，截取共同评价队列，不重新拟合。
- 几何仅取原within_cell_line（跨四剂量）750条件结果，3000基因轴固定，不取global2099或12个固定剂量组。
- 具体对比及输入根路径见execution_contract.json；以该合同和R0为准，不执行未提供的同名提示词文件中的未知内容。

## 对比分组

- E1 entity→entity_exposure、E2 entity_context→complete_metadata：exposure_addition。称暴露描述加入，包含剂量、时间、单位和模板，不解释为只改变数值剂量字段。
- C1 entity→entity_context、C2 entity_exposure→complete_metadata：context_addition_sensitivity。仍评价dose；不声称细胞系标签读出改善。
- 全部四项固定保留，不按结果改变主次。跨剂量整体几何同时包含药物、剂量及同药跨剂量关系，不等同固定剂量药物区分。

## 输出和QA

- results/r1_base_metrics.tsv、r1_paired_changes.tsv为主结果；基线读出与几何各自单列，不进入严格跨层主配对。
- 保存共同队列/排除、OOF核验、标签支持、身份hash/基因轴、NA、混淆计数和来源行号。
- 主实现用sklearn重评分；独立checker从原始OOF以手工混淆计数重算。独立检查全部72/72行及基线，不import主实现。
- 原几何数值的独立QA复用已有PASS；本轮验证其来源复制、键和轴，不声称重新独立计算所有旧RSA/NDCG。
- 无p值、跨记录相关检验、bootstrap或总分。数值零容差1e-12只用于机器精度下的方向描述，不是显著性或实际效应阈值。
- 初始运行和独立QA不覆盖旧结果；实际输入/定义错误时停止受影响分支并报告修复方案。

## 停止点

完成R1全数值、基线参照、独立QA和结果说明后停止。不进入R2、R3、R1_BROAD、C-lite、prototype、绘图或文稿修改。
