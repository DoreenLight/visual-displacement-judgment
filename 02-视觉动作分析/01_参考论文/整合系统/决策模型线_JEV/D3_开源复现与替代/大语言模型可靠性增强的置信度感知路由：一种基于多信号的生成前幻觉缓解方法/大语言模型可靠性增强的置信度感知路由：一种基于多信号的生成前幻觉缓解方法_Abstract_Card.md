# Confidence-Aware Routing for Large Language Model Reliability Enhancement: A Multi-Tier Framework

> **本卡片是摘要级定位卡，不是全文翻译。** 全文请读同目录 PDF。
> 由 `06_附录/决策模型工具/gen_jev_cards.py` 生成，请勿手改。

| 字段 | 值 |
|---|---|
| 线程 | D3 开源复现与替代 |
| 年份 | 2025 |
| 载体 | arXiv 预印本 |
| DOI | — |
| arXiv | 2510.01237 |
| 证据等级 | 第三方实测 / A |
| **编者结论** | **条件可行** |

## 链接

- 摘要页：<https://arxiv.org/abs/2510.01237>
- PDF 直链：<https://arxiv.org/pdf/2510.01237>

## 在本项目中的作用

置信度门控路由方法

## 关键数字（证据等级：A）

提出按置信度分层路由：高置信自动执行、中置信人工复核、低置信回退大模型

## 备注

★★ **本语料最值得迁移到本项目的一篇**。它不依赖 JEV 任何东西，讲的是「用校准概率做控制流分叉」这一方法论 —— 正好对上 ViKiT「代偿识别不确定时如何处置」与 AAN「什么时候介入」的实际需求。
**D4 的 arXiv 2601.04982 是这一思路在辅助机器人上的更完整实现，优先引那篇。**

## 一手来源

M, Nandakishor
