# Machine Learning-Based Real-Time Detection of Compensatory Trunk Movements Using Trunk-Wrist Inertial Measurement Units

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Jannis Gabler, Clément Lhoste, Max Quast, Laura Mayrhuber, Andrea Ronco, Olivier Lambercy, Paulius Viskaitis, Dane Donegan |
| arXiv 首次提交 | 2026 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2604.12591v1` |
| arXiv 分类 | cs.RO |
| 发展线阶段 | S5 |
| 发展线线程 | T6 |
| 康复迁移价值 | 高 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T6_物理约束·世界模型/2026_Gabler_Machine_Learning_Based_Real_Time_Detection_of.pdf` |

## 一句话定位（人工判断）

> ETH Zurich，腕+躯干双 IMU 实时检测代偿性躯干运动（XGBoost，macro-F1 0.80）；TP-03 肩肘代偿的直接基线——且它没用 Transformer，说明该任务尚未被注意力架构攻克

## 摘要（arXiv 原文照录，未翻译）

> Compensatory trunk movements (CTMs) are commonly observed after stroke and can lead to maladaptive movement patterns, limiting targeted training of affected structures. Objective, continuous detection of CTMs during therapy and activities of daily living remains challenging due to the typically complex measurements setups required, as well as limited applicability for real-time use. This study investigates whether a two-inertial measurement unit configuration enables reliable, real-time CTM detection using machine learning. Data were collected from ten able-bodied participants performing activities of daily living under simulated impairment conditions (elbow brace restricting flexion-extension, resistance band inducing flexor-synergy-like patterns), with synchronized optical motion capture (OMC) and manually annotated video recordings serving as reference. A systematic location-reduction analysis using OMC identified wrist and trunk kinematics as a minimal yet sufficient set of anatomical sensing locations. Using an extreme gradient boosting classifier (XGBoost) evaluated with leave-one-subject-out cross-validation, our two-IMU model achieved strong discriminative performance (macro-F1 = 0.80 +/- 0.07, MCC = 0.73 +/- 0.08; ROC-AUC > 0.93), with performance comparable to an OMC-based model and prediction timing suitable for real-time applications. Explainability analysis revealed dominant contributions from trunk dynamics and wrist-trunk interaction features. In preliminary evaluation using recordings from four participants with neurological conditions, the model retained good discriminative capability (ROC-AUC ~ 0.78), but showed reduced and variable threshold-dependent performance, highlighting challenges in clinical generalization. These results support sparse wearable sensing as a viable pathway toward scalable, real-time monitoring of CTMs during therapy and daily living.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2604.12591v1>
- PDF：<https://arxiv.org/pdf/2604.12591v1>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

