# Neuro-Fuzzy MSK—AAN：全文证据卡

> **不是全文翻译**，也不是临床有效性验收。原件为White Rose公开作者接受稿，带仓储封面。

## 身份与版本

- Cao Y等，*Neuro-Fuzzy Musculoskeletal Model-Driven Assist-as-Needed Control via Impedance Regulation for Rehabilitation Robots*。
- IEEE Transactions on Fuzzy Systems，33(12):4277–4288，2025；DOI `10.1109/TFUZZ.2025.3611266`。
- [原件](基于阻抗调节的神经模糊肌骨模型驱动康复机器人按需辅助控制.pdf)、[仓储记录](https://eprints.whiterose.ac.uk/id/eprint/231974/)。
- 版本：Accepted Version，不等同出版排版版。PDF页码以下含第1页仓储封面；论文自编号比PDF页码少1。

## 原文可支持的判断

| 位置 | 实际做了什么 | 本任务中的作用与边界 |
|---|---|---|
| PDF第2页 Abstract及Introduction | HFTN并行结合模糊逻辑与Transformer，估计关节力矩；FESN变量阻抗根据生理力矩估计调整刚度与阻尼，并在机器人/人主导之间适配 | 明确的“状态理解→辅助控制”整合先例；不能再把EMG估计加AAN或模糊/Transformer组合本身说成从未有人做 |
| PDF第3–7页，模型与控制方法 | HFTN力矩估计；PDFp5/Eq19监督参照为OpenSim inverse dynamics；JTCR调节FESN阻抗；PDFp6/Theorem1/Eq25以阻尼/惯性正定条件论被动性 | 模型参照不是患者净主动功真值；JTCR/模糊隶属度不是校准概率；条件性被动性不是任意估计误差/OOD下临床安全 |
| PDF第8页，V. Experimental Studies，A. Experimental Platform | 上肢机器人实验，3名健康男性参与者（24、26、29岁）；说明伦理审批 | 极小健康样本工程验证，不是卒中患者临床收益、训练剂量疗效或所有患者主动参与最大化证据 |

补核定位见[17条原件方法深核](..)和[比较说明](..)。原PDF字节及接受稿版本未变；修正章节标记不复跑原实验。

## 不能从题名或摘要补出来的结论

- 不能把“promoting active user participation”的动机/过程评价直接写成已证明ARAT/FMA改善。
- 不能因为其变量阻抗存在，就宣称已同时最优化有效练习次数、疲劳、运动质量和长期患者收益；须进一步查看具体实验变量和对照。
- 候选新方法必须正面对照这个已有框架，不能只拿单纯误差阈值当唯一对照。
