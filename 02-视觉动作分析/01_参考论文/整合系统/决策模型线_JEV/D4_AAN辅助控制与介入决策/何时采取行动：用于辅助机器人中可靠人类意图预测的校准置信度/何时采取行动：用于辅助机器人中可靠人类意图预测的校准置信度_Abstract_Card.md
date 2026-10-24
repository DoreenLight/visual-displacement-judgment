# When to Act: Calibrated Confidence for Reliable Human Intention Prediction in Assistive Robotics

> **本卡片是摘要级定位卡，不是全文翻译。** 全文请读同目录 PDF。
> 由 `06_附录/决策模型工具/gen_jev_cards.py` 生成，请勿手改。

| 字段 | 值 |
|---|---|
| 线程 | D4 AAN辅助控制与介入决策 |
| 年份 | 2026 |
| 载体 | arXiv 预印本（cs.RO） |
| DOI | — |
| arXiv | 2601.04982 |
| 证据等级 | 第三方实测 / A |
| **编者结论** | **可行主路线** |

## 链接

- 摘要页：<https://arxiv.org/abs/2601.04982>
- PDF 直链：<https://arxiv.org/pdf/2601.04982>

## 在本项目中的作用

★★★ 本语料核心：校准置信度 → ACT/HOLD 介入门控

## 关键数字（证据等级：A）

EGTEA Gaze+ 数据集：Temperature Scaling / Platt Scaling / Isotonic Regression 把 ECE 从 **≈0.40 降到 0.04**（约一个数量级）且不损失准确率。校准置信度驱动 ACT/HOLD 规则：仅在可靠性高时动作，否则拒绝介入。多模态 GRU + 后验校准 + 滞回安全门。文中明确处理震颤/共济失调等病理信号与主动运动的区分

## 备注

★★ **与本项目新方向几乎完全对应的论文，标题就是「何时行动」。**
核心主张：「错误的辅助比没有辅助更糟」（Wrong assistance can be worse than no assistance）。
**它把置信度阈值变成可解释、可调节的定量安全参数**，这正是 AAN 决策层缺的东西。
注意其自陈局限：需真实用户闭环验证；校准在病理运动等域偏移下的稳定性仍需检验 —— **对卒中患者恰好就是域偏移**。

## 一手来源

Gaus, Johannes A.、Ilg, Winfried、Haeufle, Daniel
