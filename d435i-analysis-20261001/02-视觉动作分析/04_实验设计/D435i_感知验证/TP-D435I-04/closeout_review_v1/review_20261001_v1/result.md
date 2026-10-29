# TP-D435I-04 运动质量：收尾复核结果

结论：**BLOCKED**。基于原任务包阈值，未降低要求。

保留试次 25，参与者 P001，原始 RGB/对齐深度 29,938 对，分析记录 18,310 条。排除 4 次依据用户选择，不称已证实异常。

选择索引、逐文件输入哈希、配置、脚本版本见 `input_manifest.json`。全部保留试次纳入分母，不仅报告最好片段。

## 验收项与证据

|项目|原目标|复核|结论|
|---|---|---|---|
|A/C/S解释与输入|经验证角度、代偿及任务定义|值保留NaN，各自有效性/原因明确|正式评分BLOCKED|
|数据门控|真实有效率≥95%、最大失效≤0.2s|0/25 窗满足观测门控|FAIL|
|独立判别力|AUC≥0.85、独立标签与置信区间|单人同次指令分组不满足|BLOCKED|
|延迟|采集到输出P95≤200ms|仅主机接收到输出；无设备时钟映射|BLOCKED|
|重复性|跨次重复性及置信区间|单人重复数据可描述，不能替代跨人/跨session验证|BLOCKED|
|安全监督|无执行器命令、缺失不增加助力|display-only replay，actuator_commands=[]|代码/回放边界可检查；硬件联调BLOCKED|

|试次|Q观测比例|最大无效(s)|评分门控|A状态|总体状态|
|---|---:|---:|---|---|---|
|eight_S1_20261001_131959_59d5ec91|0.749|0.103|FAIL|not_applicable|not_completed|
|eight_S1_20261001_132059_bc0d3f66|0.621|0.065|FAIL|not_applicable|not_completed|
|eight_S1_20261001_132148_b0ac2f4c|0.600|0.070|FAIL|not_applicable|not_completed|
|eight_S2_20261001_132239_ea3010e2|0.606|0.045|FAIL|not_applicable|not_completed|
|eight_S2_20261001_132326_343e7896|0.620|0.044|FAIL|not_applicable|not_completed|
|eight_S2_20261001_132413_dedfd30f|0.594|0.038|FAIL|not_applicable|not_completed|
|eight_RU_20261001_133130_b225807d|0.587|0.196|FAIL|not_scorable|not_scorable|
|eight_RU_20261001_133347_d19a29ac|0.605|0.067|FAIL|not_scorable|not_scorable|
|eight_RU_20261001_133509_ee4315e4|0.574|0.299|FAIL|not_scorable|not_scorable|
|eight_FE_20261001_133658_73162b96|0.575|0.235|FAIL|not_scorable|not_scorable|
|eight_FE_20261001_133828_a60e1ba6|0.485|0.702|FAIL|not_completed|not_completed|
|eight_FE_20261001_133920_581ea1c5|0.584|0.067|FAIL|not_scorable|not_scorable|
|eight_FE_20261001_134101_6c9daee0|0.572|1.734|FAIL|not_scorable|not_scorable|
|eight_CIRCLE_20261001_134353_ece4c4e8|0.594|0.233|FAIL|not_scorable|not_scorable|
|eight_CIRCLE_20261001_134509_b65ab28b|0.379|1.467|FAIL|not_scorable|not_scorable|
|eight_CIRCLE_20261001_134606_0e663bce|0.497|0.783|FAIL|not_scorable|not_scorable|
|eight_SHOULDER_20261001_134728_5cedac08|0.678|0.040|FAIL|not_applicable|not_completed|
|eight_SHOULDER_20261001_134819_8be00db0|0.650|0.067|FAIL|not_applicable|not_completed|
|eight_SHOULDER_20261001_134908_c025f17f|0.595|0.201|FAIL|not_applicable|not_completed|
|eight_ELBOW_20261001_135236_ac256f5f|0.622|0.068|FAIL|not_applicable|not_completed|
|eight_ELBOW_20261001_135333_1cd6edcf|0.589|0.468|FAIL|not_applicable|not_completed|
|eight_ELBOW_20261001_135439_15a01a4e|0.531|0.800|FAIL|not_applicable|not_completed|
|eight_COMBINED_20261001_135607_5bb6b817|0.582|0.769|FAIL|not_applicable|not_completed|
|eight_COMBINED_20261001_135701_cb673e7e|0.547|0.701|FAIL|not_applicable|not_completed|
|eight_COMBINED_20261001_135752_6c5fc62b|0.386|1.300|FAIL|not_applicable|not_completed|

Q是测量覆盖，不是临床运动质量。不能给静止/诱导 A=1；缺失SPARC不以周期CV或0.5补分；不设置未经验证的机器人辅助/脱开动作。旧Welch p=0.001345 不满足 p<0.001，且试次来自同一P001，不能外推独立群体临床效度。

## 复现与范围

统一命令：项目根目录 `.venv/Scripts/python.exe 04_实验设计/D435i_感知验证/TP-D435I-01/closeout_review_v1/review_four_packages.py --review-id <新的ID>`；实际命令/退出码另见 `execution.json`。

原始运行、完成的逐试次结果及冻结任务书未覆盖。本版本另存，旧综合初稿不能作为验收依据。单人单session、摆位/参考/授权类型等未提供元信息保留未知，不补造。

下一步先修正实际可用性与失效，再按原任务补参考、独立视频事件标注、跨session数据隔离及采集到输出时钟映射；质量验证和控制联调继续受各自准入约束。
