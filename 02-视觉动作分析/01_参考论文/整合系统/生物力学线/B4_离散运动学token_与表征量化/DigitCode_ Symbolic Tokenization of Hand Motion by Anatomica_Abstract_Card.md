---
title: "DigitCode: Symbolic Tokenization of Hand Motion by Anatomical Units"
thread: B4
lens: M3
relevance: "高"
year: 2026
doi: ""
arxiv_id: "2608.03127"
evidence_grade: "B（单来源确认标题，>=0.90）"
confirm_sources: "arxiv"
---

# DigitCode: Symbolic Tokenization of Hand Motion by Anatomical Units

> ⚠️ **本文件是摘要级定位卡，不是全文翻译。**
> 全部判断仅基于标题、摘要与官方元数据。**引用任何数值、方法细节或结论前，必须回原文核对。**
> 未获取原文者，该条只能作为「线索」使用，不得写入论文结论。

| 项 | 内容 |
|---|---|
| 年份 | 2026 |
| 期刊/会议 | — |
| DOI | — |
| arXiv | `2608.03127` |
| 线程 | B4 离散运动学token_与表征量化 |
| 证据范式 | M3 方法与算法（学习模型、token 化、可微优化） |
| 相关度 | 高（人工裁定） |
| 证据等级 | B（单来源确认标题，>=0.90） |
| 交叉证实来源 | arxiv |
| 官方 OA | 是（arXiv 预印本（官方 OA）） |
| 入口链接 | [https://arxiv.org/abs/2608.03127](https://arxiv.org/abs/2608.03127) |

## 摘要（来源原文，未翻译）

Hand motion carries the finest-grained information in human activity, yet the representations behind hand generation, understanding, and robot learning are overwhelmingly continuous--joint angles or MANO parameters. These are accurate but unstructured: a finger cannot be indexed or edited as a symbol, and nothing marks a pose as anatomically valid. Discrete symbolic representations supply exactly this structure, and Hand Labanotation (HL) has shown they are feasible for the hand, writing motion as a T x 40 grid of one fixed direction symbol per bone. Building on this grid, we ask the question underneath it: the anatomical unit a symbol should span--bone, finger, or whole hand. DigitCode answers it by adapting, grouping, and layering HL's alphabet along the hand's unit hierarchy within one code, cutting the symbolic representation's quantization error by three quarters. The lever is the unit, not the quantizer family: at a fixed unit, training-free and learned strong quantizers are interchangeable on reconstruction, while moving down the anatomical hierarchy is what shifts accuracy. The hierarchy also tracks what downstream tasks need. Because a finger is a genuine, enumerable unit, one per-finger token doubles as a training-free, editable handle for jobs a continuous representation cannot address--repairing malformed generated hands, and retargeting them onto robots. We release HandTok, a reproducible testbed, so hand tokenizers can be compared unit-for-unit. Project page: https://digitcode-demo.github.io.

## 为什么收录（人工裁定理由）

★★★ 【F5 撞车】DigitCode：用 Hand Labanotation 作字母表，把手部运动符号化为 T×40 网格、每骨一方向符号。「离散符号化关节表征」已被占。差异：手部/舞谱字母表/非视觉输入/无生物力学 grammar/非临床。它提出「符号该覆盖骨、指还是整手一层」的问题 —— 正好对应本项目 V6c 链拓扑，是可正面引述的方法论动机。
