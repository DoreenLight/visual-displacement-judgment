# 本轮现有数据离线重分析入口

状态：25个保留试次、29938组RGB-D已完成逐帧重推理、稳定性/轨迹/运动提示分析与画面对照。当前为工程探索结论，不修改正式验收合同。

- [三问能力与边界总报告](REPORT.md)
- [相机与算法原理及官方依据](PRINCIPLES_AND_BOUNDARIES.md)
- [下一步改进顺序](NEXT_ACTIONS.md)
- [八类覆盖汇总](analysis_20261001_v1/group_summary.json)
- [25试次指标](analysis_20261001_v1/trial_metrics.csv)、[175条关节指标](analysis_20261001_v1/joint_metrics.csv)
- [11段清晰/诊断案例与判断](analysis_20261001_v1/visual_review.json)：每段有叠加AVI、六帧对照、信号图；未进行独立完整事件标注。
- [50份腕轨迹与运动筛查CSV索引](trajectory_and_screening_exports.json)
- [750点抽样及全量帧关联检查](reconciliation_checks.json)
- [已执行汇总笔记本](analysis_companion.ipynb)
- [本轮核验记录](verification.json)
- [处理完成状态](dense_20261001_v1/progress.json)、[实际命令与模型/代码来源](dense_20261001_v1/execution.json)

## 如何继续

原始和原实时结果只读，先改基线管理、实时覆盖、小手点深度目标所属、腕角零位/轴定义和肩通道命名，再在本轮选片上前后对照。随后小批量改变视角/朝向/遮挡/速度复采，最后扩大人和session。

不要把94.8%总体离线组合可用率视为每条件都通过；不能用它宣称实时已达标。坐标是表面采样，不是已验证解剖中心；候选角不是解剖ROM；规则提示不是独立准确率。当前无法估算硬件真正最远/最近可靠距离。

## 复现

从项目根目录，使用现有.venv Python。reprocess.py需要新的--output目录，默认读取有效索引的25试次；可先用--limit与--runs小段验证。analyze.py的两个参数分别为离线数据目录和新分析目录。finalize.py生成本轮固定版本报告与回放；已有结果已封存，后续变更应使用新版本而非覆盖。

benchmark_60为试运行，preview为处理中间产物；正式本轮入口是dense_20261001_v1和analysis_20261001_v1。原片、模型推理结果及人体回放仅在本机管理，不主动上传。
