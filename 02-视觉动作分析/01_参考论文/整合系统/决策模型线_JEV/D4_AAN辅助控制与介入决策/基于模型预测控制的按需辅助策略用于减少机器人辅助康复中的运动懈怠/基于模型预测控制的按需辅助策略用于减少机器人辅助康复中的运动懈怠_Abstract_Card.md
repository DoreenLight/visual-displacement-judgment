# Model Predictive Control-Based Assist-as-Needed Strategy for Reducing Motor Slacking in Robot-Assisted Rehabilitation

> **本卡片是摘要级定位卡，不是全文翻译。** 全文请读同目录 PDF。
> 由 `06_附录/决策模型工具/gen_jev_cards.py` 生成，请勿手改。

| 字段 | 值 |
|---|---|
| 线程 | D4 AAN辅助控制与介入决策 |
| 年份 | 2026 |
| 载体 | Sensors (MDPI) 26(9): 2740（Gold OA CC-BY） |
| DOI | 10.3390/s26092740 |
| arXiv | — |
| 证据等级 | 第三方实测 / A |
| **编者结论** | **可行主路线** |

## 链接

- 摘要页：<https://www.mdpi.com/1424-8220/26/9/2740>
- PDF 直链：—

## 在本项目中的作用

★ AAN 的核心难题：motor slacking（偷懒）

## 关键数字（证据等级：A）

参与比从 0.103（不足贡献）升到 0.879（恰当贡献）；在人为将任务方向贡献压到 45% 的偷懒条件下，本文控制器的整体人类贡献比 **0.282**，显著低于误差型 AAN 基线 0.595 与遗忘因子基线 0.535

## 备注

★ **指出了误差型 AAN 的结构性缺陷**：助力和用户主动贡献被单一误差系数耦合，「主动出力减少被辅助通道吸收，伪装成小误差」。
**这正是「用置信度而非误差做介入决策」的论据** —— 误差可以伪装，参与度不能。
双通道导纳 + 凸 MPC + 能量罐保证无源性。

## 一手来源

Kim, Choonggun、Moon, Youngjin、Choi, Jaesoon
