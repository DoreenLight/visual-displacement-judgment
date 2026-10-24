# RL2026：原件身份、方法与推断边界

> **不是全文翻译**；关键方法选择核验，不是独立复现、创新定案或患者/系统安全验收。

- 题名：Personalized Adaptive Assistance With Reinforcement Learning Control Enhances Engagement, Performance, and Retention in Robot-Assisted Arm-Reaching Exercises
- DOI：`10.1109/tnsre.2025.3650096`；卷期年2026；期刊：IEEE Transactions on Neural Systems and Rehabilitation Engineering。
- [本地出版PDF](基于强化学习控制的个性化自适应助力提升机器人辅助手臂伸展训练中的参与度、表现和保持效果.pdf)；[权威页](https://research.unipd.it/handle/11577/3586318)；版本：IEEE publisher PDF, 11 file pages = journal532–542。
- 原件11页；SHA-256：`4b6e21ec3be9c4d677c6ad4d9099df49d3a8198fecf7abe990a360bf93ff97e6`；用户下载，仅获准移动一次，文件名/字节保持。
- [授权/源目标/回滚留证](..)。

## 方法、参照和适用范围

| 要核的项 | 原件可支持的判断 |
|---|---|
| 输入/输出 | 方向误差均值/峰值与历史；15健常，catch子样本6 → 阻抗g、目标权重Uc、误差阈值/难度及短时保持 |
| 反馈 | 原Fig3/Eq9–12已核：状态→RL权衡/辅助/挑战→新运动，已有联合适配 |
| 参与测量 | footnote1为身体参与代理；e_neg及平均g分别为参与/辅助代理，非心理或实际机械功真值 |
| 强度/剂量 | 4训练块，保持1/10/20min；阈值手设，不证患者剂量或长期收益 |
| 安全 | 边界参数/同意不等于患者安全；Eq12上下界符号/文字有原件歧义，不擅改实现 |
| 人群/阴性/局限 | 15健常；训练减误差控制器差异及部分过补偿/保持项未显著；无不引导视觉反馈对照 |
| 原文位置 | PDF pp.5–6/8–9，Eq9–12/Fig3/5–8/TableII–IV；S7-RL-01–13；旧16段派生留史 |

[方法及13段字面定位](..)；[7篇比较与候选裁剪](..)。物理/心理分开，不将模型/代理、健康/下肢或数理/限幅换成患者主动贡献、临床收益或一般安全保证。
