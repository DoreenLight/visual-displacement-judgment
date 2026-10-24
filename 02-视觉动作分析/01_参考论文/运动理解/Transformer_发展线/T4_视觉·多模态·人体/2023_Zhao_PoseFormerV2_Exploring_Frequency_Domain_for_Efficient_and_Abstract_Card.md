# PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Qitao Zhao, Ce Zheng, Mengyuan Liu, Pichao Wang, Chen Chen |
| arXiv 首次提交 | 2023 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2303.17472v1` |
| arXiv 分类 | cs.CV |
| 发展线阶段 | S3 |
| 发展线线程 | T4 |
| 康复迁移价值 | 中 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T4_视觉·多模态·人体/2023_Zhao_PoseFormerV2_Exploring_Frequency_Domain_for_Efficient_and.pdf` |

## 一句话定位（人工判断）

> 3D 姿态实例建模

## 摘要（arXiv 原文照录，未翻译）

> Recently, transformer-based methods have gained significant success in sequential 2D-to-3D lifting human pose estimation. As a pioneering work, PoseFormer captures spatial relations of human joints in each video frame and human dynamics across frames with cascaded transformer layers and has achieved impressive performance. However, in real scenarios, the performance of PoseFormer and its follow-ups is limited by two factors: (a) The length of the input joint sequence; (b) The quality of 2D joint detection. Existing methods typically apply self-attention to all frames of the input sequence, causing a huge computational burden when the frame number is increased to obtain advanced estimation accuracy, and they are not robust to noise naturally brought by the limited capability of 2D joint detectors. In this paper, we propose PoseFormerV2, which exploits a compact representation of lengthy skeleton sequences in the frequency domain to efficiently scale up the receptive field and boost robustness to noisy 2D joint detection. With minimum modifications to PoseFormer, the proposed method effectively fuses features both in the time domain and frequency domain, enjoying a better speed-accuracy trade-off than its precursor. Extensive experiments on two benchmark datasets (i.e., Human3.6M and MPI-INF-3DHP) demonstrate that the proposed approach significantly outperforms the original PoseFormer and other transformer-based variants. Code is released at \url{https://github.com/QitaoZhao/PoseFormerV2}.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2303.17472v1>
- PDF：<https://arxiv.org/pdf/2303.17472v1>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

