# PhysDiff: Physics-Guided Human Motion Diffusion Model

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Ye Yuan, Jiaming Song, Umar Iqbal, Arash Vahdat, Jan Kautz |
| arXiv 首次提交 | 2022 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2212.02500v3` |
| arXiv 分类 | cs.CV, cs.AI, cs.GR, cs.LG |
| 发展线阶段 | S3 |
| 发展线线程 | T6 |
| 康复迁移价值 | 中 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T6_物理约束·世界模型/2022_Yuan_PhysDiff_Physics_Guided_Human_Motion_Diffusion_Model.pdf` |

## 一句话定位（人工判断）

> 物理引导的动作扩散：把动力学可行性写进生成过程，而非事后惩罚

## 摘要（arXiv 原文照录，未翻译）

> Denoising diffusion models hold great promise for generating diverse and realistic human motions. However, existing motion diffusion models largely disregard the laws of physics in the diffusion process and often generate physically-implausible motions with pronounced artifacts such as floating, foot sliding, and ground penetration. This seriously impacts the quality of generated motions and limits their real-world application. To address this issue, we present a novel physics-guided motion diffusion model (PhysDiff), which incorporates physical constraints into the diffusion process. Specifically, we propose a physics-based motion projection module that uses motion imitation in a physics simulator to project the denoised motion of a diffusion step to a physically-plausible motion. The projected motion is further used in the next diffusion step to guide the denoising diffusion process. Intuitively, the use of physics in our model iteratively pulls the motion toward a physically-plausible space, which cannot be achieved by simple post-processing. Experiments on large-scale human motion datasets show that our approach achieves state-of-the-art motion quality and improves physical plausibility drastically (>78% for all datasets).

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2212.02500v3>
- PDF：<https://arxiv.org/pdf/2212.02500v3>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

