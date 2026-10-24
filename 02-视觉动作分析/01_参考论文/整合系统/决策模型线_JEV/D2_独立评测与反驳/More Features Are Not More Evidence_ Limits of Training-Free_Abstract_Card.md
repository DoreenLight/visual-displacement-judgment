# More Features Are Not More Evidence: Limits of Training-Free Human Activity Recognition with Jev

> **本卡片是摘要级定位卡，不是全文翻译。** 全文请读同目录 PDF。
> 由 `06_附录/决策模型工具/gen_jev_cards.py` 生成，请勿手改。

| 字段 | 值 |
|---|---|
| 线程 | D2 独立评测与反驳 |
| 年份 | 2026 |
| 载体 | arXiv 预印本 |
| DOI | — |
| arXiv | 2609.36154 |
| 证据等级 | 第三方实测 / A |
| **编者结论** | **不可行但需引证** |

## 链接

- 摘要页：<https://arxiv.org/abs/2609.36154>
- PDF 直链：<https://arxiv.org/pdf/2609.36154>

## 在本项目中的作用

★ 决定性反证：JEV 在人体活动识别上的实测天花板

## 关键数字（证据等级：A）

WISDM / UCI341 / PAMAP2 三数据集，1800 个类别均衡加速度计窗口，5400 次 JEV 决策：macro-F1 仅 0.038 / 0.118 / 0.089，监督模型 0.686–0.907。特征从 20 个扩到 32 个反而更差（预测塌向单一类）。论文明确结论：JEV 的概率「not reliably calibrated for recognition」

## 备注

★★ **本语料最重要的一篇**。Hasso Plattner Institute（波茨坦大学）独立完成，方法公开、数据集公开。
**与本项目直接相关**：ViKiT 往模型里加更多模态 token，走的正是这条「特征越多越好」的曲线，而这篇论文证明该假设在 JEV 上是反的。

## 一手来源

Konak, Orhan
