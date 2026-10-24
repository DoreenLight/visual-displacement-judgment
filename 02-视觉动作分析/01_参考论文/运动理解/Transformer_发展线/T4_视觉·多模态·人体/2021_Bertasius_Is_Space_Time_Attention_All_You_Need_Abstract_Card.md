# Is Space-Time Attention All You Need for Video Understanding?

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Gedas Bertasius, Heng Wang, Lorenzo Torresani |
| arXiv 首次提交 | 2021 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2102.05095v4` |
| arXiv 分类 | cs.CV |
| 发展线阶段 | S2 |
| 发展线线程 | T4 |
| 康复迁移价值 | 高 |
| 标题匹配相似度 | 0.909（arXiv 检索自动比对） |
| 本地 PDF | `T4_视觉·多模态·人体/2021_Bertasius_Is_Space_Time_Attention_All_You_Need.pdf` |

## 一句话定位（人工判断）

> 分解时空注意力，ViKiT encoder 直接前身

## 摘要（arXiv 原文照录，未翻译）

> We present a convolution-free approach to video classification built exclusively on self-attention over space and time. Our method, named "TimeSformer," adapts the standard Transformer architecture to video by enabling spatiotemporal feature learning directly from a sequence of frame-level patches. Our experimental study compares different self-attention schemes and suggests that "divided attention," where temporal attention and spatial attention are separately applied within each block, leads to the best video classification accuracy among the design choices considered. Despite the radically new design, TimeSformer achieves state-of-the-art results on several action recognition benchmarks, including the best reported accuracy on Kinetics-400 and Kinetics-600. Finally, compared to 3D convolutional networks, our model is faster to train, it can achieve dramatically higher test efficiency (at a small drop in accuracy), and it can also be applied to much longer video clips (over one minute long). Code and models are available at: https://github.com/facebookresearch/TimeSformer.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2102.05095v4>
- PDF：<https://arxiv.org/pdf/2102.05095v4>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

