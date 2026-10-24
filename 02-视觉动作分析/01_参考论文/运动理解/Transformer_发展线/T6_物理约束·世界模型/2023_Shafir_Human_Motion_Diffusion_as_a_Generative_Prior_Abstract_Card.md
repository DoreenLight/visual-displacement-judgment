# Human Motion Diffusion as a Generative Prior

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Yonatan Shafir, Guy Tevet, Roy Kapon, Amit H. Bermano |
| arXiv 首次提交 | 2023 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2303.01418v3` |
| arXiv 分类 | cs.CV, cs.GR |
| 发展线阶段 | S3 |
| 发展线线程 | T6 |
| 康复迁移价值 | 中 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T6_物理约束·世界模型/2023_Shafir_Human_Motion_Diffusion_as_a_Generative_Prior.pdf` |

## 一句话定位（人工判断）

> 运动学/动力学约束的扩散先验

## 摘要（arXiv 原文照录，未翻译）

> Recent work has demonstrated the significant potential of denoising diffusion models for generating human motion, including text-to-motion capabilities. However, these methods are restricted by the paucity of annotated motion data, a focus on single-person motions, and a lack of detailed control. In this paper, we introduce three forms of composition based on diffusion priors: sequential, parallel, and model composition. Using sequential composition, we tackle the challenge of long sequence generation. We introduce DoubleTake, an inference-time method with which we generate long animations consisting of sequences of prompted intervals and their transitions, using a prior trained only for short clips. Using parallel composition, we show promising steps toward two-person generation. Beginning with two fixed priors as well as a few two-person training examples, we learn a slim communication block, ComMDM, to coordinate interaction between the two resulting motions. Lastly, using model composition, we first train individual priors to complete motions that realize a prescribed motion for a given joint. We then introduce DiffusionBlending, an interpolation mechanism to effectively blend several such models to enable flexible and efficient fine-grained joint and trajectory-level control and editing. We evaluate the composition methods using an off-the-shelf motion diffusion model, and further compare the results to dedicated models trained for these specific tasks.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2303.01418v3>
- PDF：<https://arxiv.org/pdf/2303.01418v3>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

