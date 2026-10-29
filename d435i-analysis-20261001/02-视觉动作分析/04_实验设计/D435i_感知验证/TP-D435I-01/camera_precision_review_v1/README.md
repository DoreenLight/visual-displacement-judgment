---
type: evidence_review
revision: "1.0"
updated: 2026-10-01
stage: exploratory_engineering
status: historical_data_recomputed_and_archived
hardware: RealSense D435i
measurement_session: ended_by_user
---

# 历史相机精度证据补充与人体轨迹、角度范围复核

**本轮人体测量已按用户要求结束，第5组没有采集。** 本次只读检索 `01-摄像头`，不启动历史相机脚本、不另开相机、不修改相机资料库、既有原始数据或正式四问任务包。

阅读顺序：[结论与证据报告](REPORT.md) → [角度换算与计数边界](ANGLE_CONVERSION.md) → [数值复算](historical_fit_recomputed.json)。

## 当前结论

1. 历史记录确认同一序列号 D435i 在约20–22cm、固定白纸刚性目标、区域平均条件下具备很好的Z向静止重复性和端点尺度一致性。四点JSON可复算；不是人体各关节点三维动态准确性证明。
2. 本轮确认静止的30秒人体输出：肩/肘/腕位置波动P95为4.92/8.00/6.67mm，各轴SD约0.77–3.31mm。只能支持当前布置下单段静止稳定性初验，不能承诺每个关节点连续轨迹的误差上界。
3. 用户报告长度为上臂300mm、前臂290mm、腕白点到中指根部112mm。以固定旋转中心和纯旋转为条件，2mm端点弦长对应0.382°/0.395°/1.023°，5mm对应0.955°/0.988°/2.558°。不是已验证的最小可检测/可计数角。
4. 当前程序计数的最小可设单侧门槛仍为5°，对称往返须跨两侧极值；没有更改软件门槛。历史低MDR不等于程序能自动计数0.1°动作。

## 本轮交付

| 文件 | 内容 |
|---|---|
| [REPORT.md](REPORT.md) | 历史证据、人体逐点稳定性、适用距离和失效边界、四组端点对照最新口径 |
| [ANGLE_CONVERSION.md](ANGLE_CONVERSION.md) | 弦长、投影、误差传播公式与个人长度示例；运动角、角误差、计数门槛分开 |
| [subject_lengths.json](subject_lengths.json) | 用户提供长度及表面点定义；不是独立骨骼中心标定 |
| [historical_fit_recomputed.json](historical_fit_recomputed.json)、[步进差值](historical_steps.csv) | 四点拟合及真实已记录步进的重算 |
| [个人换算表](personal_angle_conversion.csv)、[敏感性示例](angle_sensitivity_examples.csv) | 可复核几何数值，不标为相机实测角精度 |
| [假设误差传播](conditional_error_propagation.json)、[计数几何](counting_geometry.json) | 假设每点误差界与软件门槛对应位移量级 |
| [模型段长](model_lengths_for_context.json) | 本次模型点间长度，仅说明点定义差异，不替代人工长度 |
| [四组端点复核](endpoint_pairs_reviewed.json) | 1–3原端点；4新端点且用户确认仍为50mm |
| [analyze.py](analyze.py)、[复算notebook](reproduce.ipynb) | 只读输入的重算入口，不接相机 |
| [来源清单](source_manifest.json)、[数值验证](verification.json)、[归档核验](archive_verification.json) | 100个相机目录文件SHA-256、公式/重算及链接/冻结资料核对 |

## 复现

在 `02-视觉动作分析` 根目录运行：

```powershell
.\.venv\Scripts\python.exe '04_实验设计/D435i_感知验证/TP-D435I-01/camera_precision_review_v1/analyze.py'
```

程序只读取历史数据、本次已完成坐标及用户长度，重算派生表。不运行SDK相机采集，不回填人体准确性值。相机文件夹完整清单仅用于检索与漂移留证；实际数值依据为报告中明确列出的JSON、报告和坐标记录，并非100个文件都含精度试验数据。

原始数据、早先计算及澄清过程分别保留；当前端点口径见报告，旧的“距离待确认”及“约16mm判断撤回”是澄清中的阶段状态，不再作为现行结论。

## 下一步

本次数据汇总与个人长度几何换算已完成。将来如继续研究，需要匹配同一物理点、在实际距离和姿态下评估动态参考、重复性、同步及角度轴。是否重新采集由用户另行安排；不会依据此前第5组待就位问题恢复采集。原四问任务包正式结论和验收阈值保持原记录。
