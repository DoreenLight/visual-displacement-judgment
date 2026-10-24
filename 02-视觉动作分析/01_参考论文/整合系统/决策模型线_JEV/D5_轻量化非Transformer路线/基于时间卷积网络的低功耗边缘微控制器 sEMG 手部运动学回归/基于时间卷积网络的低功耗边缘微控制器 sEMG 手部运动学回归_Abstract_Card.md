# sEMG-based Regression of Hand Kinematics with Temporal Convolutional Networks on a Low-Power Edge Microcontroller

> **本卡片是摘要级定位卡，不是全文翻译。** 全文请读同目录 PDF。
> 由 `06_附录/决策模型工具/gen_jev_cards.py` 生成，请勿手改。

| 字段 | 值 |
|---|---|
| 线程 | D5 轻量化非Transformer路线 |
| 年份 | 2021 |
| 载体 | IEEE（EMBC 附近） |
| DOI | 10.1109/EMBC.46406.2021.9630825 |
| arXiv | — |
| 证据等级 | 第三方实测 / A |
| **编者结论** | **可行主路线** |

## 链接

- 摘要页：<https://ieeexplore.ieee.org/document/9524188>
- PDF 直链：—

## 在本项目中的作用

超低功耗微控制器上的 sEMG 回归

## 关键数字（证据等级：A）

NinaPro DB8，12 名被试 9 个对侧动作，5 个连续自由度：MAE 6.89°（比 SoA 好 0.15°）；int8 量化后仅 **70.9 kB**；GAP8 上 4.76 ms 延迟、0.243 mJ/次推理

## 备注

**手部 sEMG → 腕/手指运动学**，与本项目腕部任务最接近的轻量化先例。
70.9 kB 与任何十亿级参数模型不在一个量级 —— **这说明「非自回归」不是轻量化的答案，「小模型才是」。**
注意：它做的是回归（连续角度），不是介入决策，恰好印证 D2 的分工判断。

## 一手来源

Grenzholt 等（TEMPONet）
