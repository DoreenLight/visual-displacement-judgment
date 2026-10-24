# A Human-in-the-Loop Confidence-Aware Failure Recovery Framework for Modular Robot Policies

> **本卡片是摘要级定位卡，不是全文翻译。** 全文请读同目录 PDF。
> 由 `06_附录/决策模型工具/gen_jev_cards.py` 生成，请勿手改。

| 字段 | 值 |
|---|---|
| 线程 | D4 AAN辅助控制与介入决策 |
| 年份 | 2026 |
| 载体 | ACM/IEEE HRI 2026, pp. 346-355（DOI 10.1145/3757279.3788668） |
| DOI | 10.1145/3757279.3788668 |
| arXiv | 2602.10289 |
| 证据等级 | 第三方实测 / A |
| **编者结论** | **可行主路线** |

## 链接

- 摘要页：<https://arxiv.org/abs/2602.10289>
- PDF 直链：<https://arxiv.org/pdf/2602.10289>

## 在本项目中的作用

★★ 同行评审：置信感知 + 人类负荷成本决定问谁、何时问

## 关键数字（证据等级：A）

在机器人辅助进食系统上部署，真实行动障碍用户家庭研究：提升失效恢复成功率同时降低用户查询负荷。两个决策分离：模块选择器（该问哪个模块）+ 查询算法（是否该问人）

## 备注

★★ **本语料证据等级最高的一篇**（同行评审会议论文，非预印本）。
**关键设计是「两个决策分离」**：问哪个模块 vs 是否要问人。
迁移到 AAN：把「是否介入」与「介入哪个自由度/哪个方向」拆成两个独立校准决策 —— 这比单阈值更贴合上肢腕-肘-肩的多关节结构。

## 一手来源

Banerjee, Rohan、Palempalli, Krishna、Yang, Bohan、Fang, Jiaying、Abdullah, Alif、Silver, Tom、Dean, Sarah、Bhattacharjee, Tapomayukh
