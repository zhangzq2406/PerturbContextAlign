# R2执行合同（数值读取和计算前固定）

仅执行20260925用户授权的R2。R0来源、缓存、轴和依赖直接复用，不重做总体审计；R1已验收且不重跑。具体输入根路径和参数见execution_contract.json。

## 问题与评价单位

固定细胞系、剂量后，两次重复测量的药物响应成对关系是否一致？一次重复的排序能否恢复另一次重复的响应邻域？十二组，每组187或188条件；各细胞系使用原3000基因轴。原2250 primary条件全部保留，仅沿用六个原YM155排除；不按可靠性或响应幅度筛选。

## 计算

1. 从原float32 measurement均值缓存转float64，分别计算rep1/rep2的log2((treated+1)/(control_B+1))。不用两个重复logFC的平均替代原合并均值效应。
2. 分别构建未中心化cosine矩阵；相同上三角（去掉对角线）的average-rank Spearman为几何重复一致性。常数几何NA，无pair-level p值。
3. k=10，query排除自身。truth精确并列处按剩余top10名额等分线性relevance，relevance总和10；prediction精确tie块的DCG为块内平均relevance乘其前10位折损和。IDCG由fractional relevance排序得到。
4. rep1→rep2和rep2→rep1分别计算，箭头左边为预测排序，右边为truth。每个方向减去自身truth的解析随机排序期望：mean(relevance)×sum(discounts)/IDCG。excess为差值，不是比值或校正值。
5. 所有query和原始分母保留；truth候选全相同为UNINFORMATIVE_TRUTH、分数NA；候选不足按原规则NA。非有限/负均值停止受影响分析；零效应范数按旧runner规则停止受影响组，不能静默设cosine0或删条件。

## 对照、信息权限、统计边界

匹配字段为cell_line/time/replicate/plate。核验rep1/rep2 B组互斥、同一重复内控制组跨药物复用；记录共享关系，不把条件对、两个方向或十二组视为独立生物重复。缓存级计算不重新聚合原细胞；原float32精度仍是限制。只描述完整十二组，不做显著性、bootstrap、回归、噪声解释比例或可靠性除法校正。

R2不是严格noise ceiling，也不是独立研究复制。其固定剂量范围不同于R1每条细胞系750条件的跨剂量几何，不能作为R1的校正分母或噪声归因。R1中暴露加入对读出与局部邻域的同步改善必须保留。

## 输出与验收

保存输入hash清单、十二组队列/基因轴manifest、12条几何汇总、24条双方向汇总、4500条query、完整条件去向、共享对照表和重复几何NPZ。主实现复用冻结metric helper，独立QA不读或import主实现/helper，从同一原缓存另算全部组，检查ties/NA/随机参照的独立小例子；数值容差1e-10，精确tie定义不使用容差合并。

验收要求原成员顺序和3000基因轴匹配、12组/两个方向无遗漏、旧输入hash不变、所有NA/分母保留、独立QA PASS。只用CPU，目标峰值<2GiB；不训练、不新编码、不改原split、不读raw H5AD、不作正式图或改稿。发现真实错误先保留证据、停止受影响分支，单独提出修复方案，不覆盖旧结果。

完成R2与独立QA后停止，不进入R3、R1_BROAD、C-lite或prototype。

## 数值读取前收到的提示词补充

用户随后提供了${AUTHOR_HOME}/Downloads/codex_R2_response_geometry_reproducibility_20260925.md，已完整读取并核对，数值计算尚未开始。增加logs/、事先固定的无并列正确排序/预测全并列/truth全并列/第10位并列/候选不足/共同条件与基因轴重排测试。另固定复制十二个同组complete_metadata/source_name的六编码器72行和三基线36行，分别保存；它们的truth为原合并效应，不是分重复效应，只作同组并列参照。若无法精确绑定，只停止这项参照，不捏造数据或阻断独立R2核心结果。
