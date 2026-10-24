# Personalization of Wearable Sensor-Based Joint Kinematic Estimation Using Computer Vision for Hip Exoskeleton Applications

> **本卡片是摘要级定位卡，不是全文翻译。** 全文请读同目录 PDF。
> 由 `06_附录/决策模型工具/gen_jev_cards.py` 生成，请勿手改。

| 字段 | 值 |
|---|---|
| 线程 | D5 轻量化非Transformer路线 |
| 年份 | 2024 |
| 载体 | arXiv 预印本（后有会议版） |
| DOI | — |
| arXiv | 2411.15366 |
| 证据等级 | 第三方实测 / A |
| **编者结论** | **可行主路线** |

## 链接

- 摘要页：<https://arxiv.org/abs/2411.15366>
- PDF 直链：<https://arxiv.org/pdf/2411.15366>

## 在本项目中的作用

TCN 的跨被试个性化迁移（直击本项目已知难点）

## 关键数字（证据等级：A）

仅需 1–2 个步态周期的数据做迁移学习，RMSE 较纯 TCN 降低 9.7% / 19.9%

## 备注

**直击本项目已知难点**（跨被试泛化只有 50–70%）。用计算机视觉姿态估计产生的标签 + TCN 迁移，是绕开「每个新被试都要重新标定」的可行路径。
**且它同时命中了 AAN 决策层的一个前置问题**：决策阈值如果因被试而异，就必须先解决个体化，否则阈值不可迁移。

## 一手来源

Song, Changseob、Ivanyuk-Skulskyi, Bogdan、Krieger, Adrian、Luo, Kaitao、Kang, Inseung
