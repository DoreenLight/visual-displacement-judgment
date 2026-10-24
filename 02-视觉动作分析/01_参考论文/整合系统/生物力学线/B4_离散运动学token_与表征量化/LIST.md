# B4 离散运动学token_与表征量化 — 论文清单

> 本线程目的：用户已决定「连续回归主头 + 离散 kinematic token 辅助头」。本线程要回答：离散词表怎么建才能有解剖含义、codebook 会不会塌缩、可解释性如何验证。也是与 TokenHMR/MotionBERT 划清边界的依据。

> 共 11 篇（已交叉证实可引用 11 篇，其中相关度「高」5 篇；官方 OA 可取 10 篇）。
> 本清单只收录**已交叉证实**条目；待裁定条目见 [`../index.json`](../index.json) 的 `pending_adjudication`。

## 清单

| # | 论文 | 年 | 期刊/会议 | DOI / arXiv | 证据范式 | 相关度 | 定位卡 | OA |
|---|---|---|---|---|---|---|---|---|
| 1 | [Masked reconstruction model of latent space vector quantization for human skeleton-based action recognition](../B4_离散运动学token_与表征量化/Masked reconstruction model of latent space vector quantizat_Abstract_Card.md) | 2026 | Neurocomputing | `10.1016/j.neucom.2025.132126` | M3 方法与算法（学习模型、token 化、可微优化） | 高 | ✅ | — |
| 2 | [DigitCode: Symbolic Tokenization of Hand Motion by Anatomical Units](../B4_离散运动学token_与表征量化/DigitCode_ Symbolic Tokenization of Hand Motion by Anatomica_Abstract_Card.md) | 2026 | — | `2608.03127` | M3 方法与算法（学习模型、token 化、可微优化） | 高 | ✅ | ✅ |
| 3 | [Kinematic Tokenization: Optimization-Based Continuous-Time Tokens for Learnable Decision Policies in Noisy Time Series](../B4_离散运动学token_与表征量化/Kinematic Tokenization_ Optimization-Based Continuous-Time T_Abstract_Card.md) | 2026 | — | `2601.09949` | M3 方法与算法（学习模型、token 化、可微优化） | 低 | ✅ | ✅ |
| 4 | [Fine-grained Motion Retrieval via Joint-Angle Motion Images and Token-Patch Late Interaction](../B4_离散运动学token_与表征量化/Fine-grained Motion Retrieval via Joint-Angle Motion Images_Abstract_Card.md) | 2026 | — | `2603.09930` | M3 方法与算法（学习模型、token 化、可微优化） | 中 | ✅ | ✅ |
| 5 | [Multi-View 3D Human Pose Estimation with Weakly Synchronized Images](../B4_离散运动学token_与表征量化/Multi-View 3D Human Pose Estimation with Weakly Synchronized_Abstract_Card.md) | 2025 | AAAI Conference on Artificial Intelligence | `10.1609/aaai.v39i5.32511` | M3 方法与算法（学习模型、token 化、可微优化） | 高 | ✅ | ✅ |
| 6 | [VQ-HPS: Human Pose and Shape Estimation in a Vector-Quantized Latent Space](../B4_离散运动学token_与表征量化/VQ-HPS_ Human Pose and Shape Estimation in a Vector-Quantize_Abstract_Card.md) | 2024 | Lecture notes in computer science | `10.1007/978-3-031-72943-0_27` | M3 方法与算法（学习模型、token 化、可微优化） | 高 | ✅ | ✅ |
| 7 | [BEST: BERT Pre-training for Sign Language Recognition with Coupling Tokenization](../B4_离散运动学token_与表征量化/BEST_ BERT Pre-training for Sign Language Recognition with C_Abstract_Card.md) | 2023 | Proceedings of the AAAI Conference on Artifi | `10.1609/aaai.v37i3.25470` | M3 方法与算法（学习模型、token 化、可微优化） | 低 | ✅ | ✅ |
| 8 | [Learning Joint Twist Rotation for 3D Human Pose Estimation from a Single Image](../B4_离散运动学token_与表征量化/Learning Joint Twist Rotation for 3D Human Pose Estimation f_Abstract_Card.md) | 2021 | Proceedings of the 16th International Joint  | `10.5220/0010222703790386` | M3 方法与算法（学习模型、token 化、可微优化） | 中 | ✅ | ✅ |
| 9 | [Robot Motion Planning in Learned Latent Spaces](../B4_离散运动学token_与表征量化/Robot Motion Planning in Learned Latent Spaces_Abstract_Card.md) | 2019 | IEEE Robotics and Automation Letters | `10.1109/lra.2019.2901898` | M3 方法与算法（学习模型、token 化、可微优化） | 中 | ✅ | ✅ |
| 10 | [Learning 3D Human Pose from Structure and Motion](../B4_离散运动学token_与表征量化/Learning 3D Human Pose from Structure and Motion_Abstract_Card.md) | 2018 | Lecture notes in computer science | `10.1007/978-3-030-01240-3_41` | M3 方法与算法（学习模型、token 化、可微优化） | 中 | ✅ | ✅ |
| 11 | [Learning Pose Grammar to Encode Human Body Configuration for 3D Pose Estimation](../B4_离散运动学token_与表征量化/Learning Pose Grammar to Encode Human Body Configuration for_Abstract_Card.md) | 2017 | — | `1710.06513` | M3 方法与算法（学习模型、token 化、可微优化） | 高 | ✅ | ✅ |

## 证据等级分布

| 等级 | 含义 | 条数 |
|---|---|---|
| A | ≥2 个独立来源确认标题 | 6 |
| B | 单来源确认标题（≥0.90） | 5 |

> 相关度、证据范式、线程归属**均为人工裁定**，不是来源官方字段，已在 index.json 中与官方元数据分列。
