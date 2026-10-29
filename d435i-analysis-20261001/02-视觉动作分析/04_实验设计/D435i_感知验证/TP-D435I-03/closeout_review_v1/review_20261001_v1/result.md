# TP-D435I-03 肩肘代偿：收尾复核结果

结论：**BLOCKED**。基于原任务包阈值，未降低要求。

保留试次 25，参与者 P001，原始 RGB/对齐深度 29,938 对，分析记录 18,310 条。排除 4 次依据用户选择，不称已证实异常。

选择索引、逐文件输入哈希、配置、脚本版本见 `input_manifest.json`。全部保留试次纳入分母，不仅报告最好片段。

## 验收项与证据

|项目|原目标|现有证据|结论|
|---|---|---|---|
|肩、肘检测性能|独立测试集各敏感度/特异度≥0.90|没有双人独立标签及隔离测试集|BLOCKED|
|组合混淆/一致性|独立标签、计数、置信区间|只有指令条件及观测分布|BLOCKED|
|事件检测延迟|可观测事件P95≤300ms|没有独立事件起点与同步映射|BLOCKED|
|可判读覆盖|候选≥95%，unknown保留|见每次observable_rate|逐试次见CSV|

|试次|指令条件（非真值）|可判读覆盖|unknown帧|众数观测|报警占可判读比例|试次开始到首警(s)|
|---|---|---:|---:|---|---:|---:|
|eight_S1_20261001_131959_59d5ec91|not_applicable|0.749|226|normal|0.000|不可计算|
|eight_S1_20261001_132059_bc0d3f66|not_applicable|0.621|341|combined|1.000|0.104|
|eight_S1_20261001_132148_b0ac2f4c|not_applicable|0.600|360|combined|1.000|0.071|
|eight_S2_20261001_132239_ea3010e2|not_applicable|0.000|900|unknown|不可计算|不可计算|
|eight_S2_20261001_132326_343e7896|not_applicable|0.620|342|normal|0.000|不可计算|
|eight_S2_20261001_132413_dedfd30f|not_applicable|0.594|365|normal|0.000|不可计算|
|eight_RU_20261001_133130_b225807d|normal|0.587|1070|normal|0.443|40.565|
|eight_RU_20261001_133347_d19a29ac|normal|0.605|571|shoulder_compensation|0.985|0.098|
|eight_RU_20261001_133509_ee4315e4|normal|0.574|480|combined|0.937|0.087|
|eight_FE_20261001_133658_73162b96|normal|0.575|683|normal|0.006|53.350|
|eight_FE_20261001_133828_a60e1ba6|normal|0.485|364|elbow_compensation|0.913|0.101|
|eight_FE_20261001_133920_581ea1c5|normal|0.584|773|shoulder_compensation|1.000|0.099|
|eight_FE_20261001_134101_6c9daee0|normal|0.572|1218|combined|0.955|0.106|
|eight_CIRCLE_20261001_134353_ece4c4e8|normal|0.594|575|normal|0.000|不可计算|
|eight_CIRCLE_20261001_134509_b65ab28b|normal|0.379|506|normal|0.146|0.098|
|eight_CIRCLE_20261001_134606_0e663bce|normal|0.497|367|shoulder_compensation|0.893|2.813|
|eight_SHOULDER_20261001_134728_5cedac08|shoulder_compensation|0.678|312|shoulder_compensation|0.891|0.076|
|eight_SHOULDER_20261001_134819_8be00db0|shoulder_compensation|0.650|325|combined|1.000|0.087|
|eight_SHOULDER_20261001_134908_c025f17f|shoulder_compensation|0.595|375|combined|0.969|0.122|
|eight_ELBOW_20261001_135236_ac256f5f|elbow_compensation|0.622|362|combined|0.988|0.103|
|eight_ELBOW_20261001_135333_1cd6edcf|elbow_compensation|0.589|473|combined|0.931|0.098|
|eight_ELBOW_20261001_135439_15a01a4e|elbow_compensation|0.531|704|combined|0.812|1.723|
|eight_COMBINED_20261001_135607_5bb6b817|combined|0.582|387|combined|0.827|0.081|
|eight_COMBINED_20261001_135701_cb673e7e|combined|0.547|433|combined|0.828|0.312|
|eight_COMBINED_20261001_135752_6c5fc62b|combined|0.386|658|combined|0.671|0.083|

首警时间以实际主机输出时间减试次开始，仅是流程观测，不是独立事件起点的检测延迟。静止不视作完成腕训练的normal真值。不能用试次众数为normal证明整次零虚警。没有独立标签时不发布灵敏度、特异度或准确率。

## 复现与范围

统一命令：项目根目录 `.venv/Scripts/python.exe 04_实验设计/D435i_感知验证/TP-D435I-01/closeout_review_v1/review_four_packages.py --review-id <新的ID>`；实际命令/退出码另见 `execution.json`。

原始运行、完成的逐试次结果及冻结任务书未覆盖。本版本另存，旧综合初稿不能作为验收依据。单人单session、摆位/参考/授权类型等未提供元信息保留未知，不补造。

下一步先修正实际可用性与失效，再按原任务补参考、独立视频事件标注、跨session数据隔离及采集到输出时钟映射；质量验证和控制联调继续受各自准入约束。
