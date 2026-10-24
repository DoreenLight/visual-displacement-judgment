# D6 康复侧既有实证

> 3 篇 ｜ ✅ 已入库 3 ｜ 需人工取件 0
> 本文件由 `build_jev.py` 生成。数字口径见 `../index.json`。

| # | 年 | 论文题名 | 载体 | 链接 | 证据等级 | 编者结论 | 本地 |
|---|---|---|---|---|---|---|---|
| 1 | 2008 | A Real-Time, 3-D Musculoskeletal Model for Dynamic Simulation of Arm Movements | IEEE TBME | [DOI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2971671) | 第三方实测 | 仅作背景 | ✅ |
| 2 | 2026 | More Features Are Not More Evidence（康复视角摘录） | arXiv 预印本（与 D2 同篇） | [2609.36154](https://arxiv.org/abs/2609.36154) | 第三方实测 | 不可行但需引证 | ✅ |
| 3 | 2026 | Markerless On-Device Detection of Compensatory Movement Patterns in Upper-Limb Rehabilitation Exercises from Monocular RGB Video | Sensors (MDPI) | [DOI](https://doi.org/10.3390/s26165054) | 第三方实测 | 仅作背景 | ✅ |

---

### A Real-Time, 3-D Musculoskeletal Model for Dynamic Simulation of Arm Movements（2008，IEEE TBME）

- **作者**：（见语料三索引）
- **来源**：<https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2971671>
- **关键数字（A）**：待人工取件：PMC 落地页可达，PDF 端点有 reCAPTCHA
- **证据等级**：第三方实测
- **编者结论**：仅作背景
- **本地工件**：PDF 全文已在库：`01_参考论文/整合系统/生物力学线/B2_肌肉骨骼动力学_与可微仿真/A Real-Time, 3-D Musculoskeletal Model for Dynamic Simulatio.pdf`（跨库引用，不重复下载）
- **备注**：**语料三待取件清单 A 组条目**，交叉索引。
其三维肌肉骨骼模型可为 D4 的介入决策提供「本次需要的力矩是多少」这一物理量。

### More Features Are Not More Evidence（康复视角摘录）（2026，arXiv 预印本（与 D2 同篇））

- **作者**：Konak, Orhan
- **来源**：<https://arxiv.org/abs/2609.36154>
- **关键数字（A）**：三个数据集 WISDM / UCI341 / PAMAP2 均为可穿戴加速度计人体活动识别标准基准
- **证据等级**：第三方实测
- **编者结论**：不可行但需引证
- **本地工件**：PDF 全文已在库：`01_参考论文/整合系统/决策模型线_JEV/D2_独立评测与反驳/More Features Are Not More Evidence_ Limits of Training-Free.pdf`（跨库引用，不重复下载）
- **备注**：**交叉索引**，PDF 见 D2。此处只标注它在康复语境下的意义：目前唯一一篇把 JEV 放到人体活动识别任务上做的公开评测。

### Markerless On-Device Detection of Compensatory Movement Patterns in Upper-Limb Rehabilitation Exercises from Monocular RGB Video（2026，Sensors (MDPI)）

- **作者**：（见语料三索引）
- **来源**：<https://doi.org/10.3390/s26165054>
- **关键数字（A）**：端侧笔记本 22–30 fps，macro-F1 0.75
- **证据等级**：第三方实测
- **编者结论**：仅作背景
- **本地工件**：PDF 全文已在库：`01_参考论文/运动理解/Transformer_发展线/T6_物理约束·世界模型/2026_端侧单目RGB_上肢代偿模式检测.pdf`（跨库引用，不重复下载）
- **备注**：**语料三已入库**，此处交叉索引不重复收 PDF。意义：康复代偿识别这个任务上，端侧卷积路线已能到 macro-F1 0.75。
**其判别输出正是 D4 介入决策层的天然输入之一**（代偿出现与否）。
