# Shoulder2025：原件身份、方法与推断边界

> **不是全文翻译**；关键方法选择核验，不是独立复现、创新定案或患者/系统安全验收。

- 题名：Assist-as-Needed Control of a Shoulder Rehabilitation Robot Using a Virtual Biomechanical Model and Online Stiffness Adaptation
- DOI：`10.1109/tnsre.2025.3579533`；卷期年2025；期刊：IEEE Transactions on Neural Systems and Rehabilitation Engineering。
- [本地出版PDF](基于虚拟生物力学模型和在线刚度自适应的肩部康复机器人按需辅助控制.pdf)；[权威页](https://doi.org/10.1109/tnsre.2025.3579533)；版本：IEEE publisher PDF,10 file pages=journal2358–2367。
- 原件10页；SHA-256：`e71f6fb2dfc06eb87118b3056e4e82e6bc6dc67fb0554698ba3a008ceabf1768`；用户下载，仅获准移动一次，文件名/字节保持。
- [授权/源目标/回滚留证](..)。

## 方法、参照和适用范围

| 要核的项 | 原件可支持的判断 |
|---|---|
| 输入/输出 | 模型/机器人运动学、交互力；VBSRM逆动力学期望人体力矩 → β混合力矩、梯度刚度、PID与ADL跟踪/峰值力矩 |
| 反馈 | 模型→混合AAN/在线刚度→新观察已有，不是首次肌骨辅助结合 |
| 参与测量 | 模型期望人体力矩不等于独立自主力矩；β非校准参与概率，无心理量表 |
| 强度/剂量 | ADL追踪/参数适配，未确具体动作或重复数，不证明患者剂量 |
| 安全 | III.D稳定讨论及β/关节/力矩限幅、急停/监督；非一般被动性/估计误差/OOD安全证明 |
| 人群/阴性/局限 | 10健常7男3女，22–38；没有患者临床结论；环境/长期适配未全面测试 |
| 原文位置 | PDF pp.6–7/9，TableII–III/III.D/V局限；S7-SH-01–08 |

[方法及8段字面定位](..)；[7篇比较与候选裁剪](..)。物理/心理分开，不将模型/代理、健康/下肢或数理/限幅换成患者主动贡献、临床收益或一般安全保证。
