# Assistance level quantification-based human-robot interaction space reshaping for rehabilitation training

> **本卡片是摘要级定位卡，不是全文翻译。** 全文请读同目录 PDF。
> 由 `06_附录/决策模型工具/gen_jev_cards.py` 生成，请勿手改。

| 字段 | 值 |
|---|---|
| 线程 | D4 AAN辅助控制与介入决策 |
| 年份 | 2023 |
| 载体 | Frontiers in Neurorobotics 17: 1161007（开放获取） |
| DOI | 10.3389/fnbot.2023.1161007 |
| arXiv | — |
| 证据等级 | 第三方实测 / A |
| **编者结论** | **可行主路线** |

## 链接

- 摘要页：<https://journal.frontiersin.org/article/10.3389/fnbot.2023.1161007>
- PDF 直链：—

## 在本项目中的作用

AAN 的能力评估：EMG + 运动学双输入分类

## 关键数字（证据等级：A）

按恢复阶段设计七套实验协议区分训练进程；用 PSO-SVM 分类模型 + LSTM-KF 回归模型，从 sEMG 与运动学数据识别患者运动能力，驱动 AAN 控制与交互空间重塑

## 备注

**已有人把 sEMG 接到 AAN 决策上**，用的正是本项目关心的 sEMG 通路。
其局限：SVM/KF 不是校准过的概率，分类器输出不能直接当安全阈值用 —— 这正是 D4 主张「必须自己校准」的现实佐证。

## 一手来源

Li, Xiangyun、Chen, Pengcheng、Gong, Shan、Yu, Xi、He, Hongchen、Li, Kang
