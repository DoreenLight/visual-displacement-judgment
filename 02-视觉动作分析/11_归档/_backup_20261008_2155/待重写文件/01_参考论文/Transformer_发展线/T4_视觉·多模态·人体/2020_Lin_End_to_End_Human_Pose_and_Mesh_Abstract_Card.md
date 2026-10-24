# End-to-End Human Pose and Mesh Reconstruction with Transformers

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Kevin Lin, Lijuan Wang, Zicheng Liu |
| arXiv 首次提交 | 2020 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2012.09760v3` |
| arXiv 分类 | cs.CV |
| 发展线阶段 | S1 |
| 发展线线程 | T4 |
| 康复迁移价值 | 中 |
| 标题匹配相似度 | 0.955（arXiv 检索自动比对） |
| 本地 PDF | `T4_视觉·多模态·人体/2020_Lin_End_to_End_Human_Pose_and_Mesh.pdf` |

## 一句话定位（人工判断）

> 早期 image→mesh Transformer

## 摘要（arXiv 原文照录，未翻译）

> We present a new method, called MEsh TRansfOrmer (METRO), to reconstruct 3D human pose and mesh vertices from a single image. Our method uses a transformer encoder to jointly model vertex-vertex and vertex-joint interactions, and outputs 3D joint coordinates and mesh vertices simultaneously. Compared to existing techniques that regress pose and shape parameters, METRO does not rely on any parametric mesh models like SMPL, thus it can be easily extended to other objects such as hands. We further relax the mesh topology and allow the transformer self-attention mechanism to freely attend between any two vertices, making it possible to learn non-local relationships among mesh vertices and joints. With the proposed masked vertex modeling, our method is more robust and effective in handling challenging situations like partial occlusions. METRO generates new state-of-the-art results for human mesh reconstruction on the public Human3.6M and 3DPW datasets. Moreover, we demonstrate the generalizability of METRO to 3D hand reconstruction in the wild, outperforming existing state-of-the-art methods on FreiHAND dataset. Code and pre-trained models are available at https://github.com/microsoft/MeshTransformer.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2012.09760v3>
- PDF：<https://arxiv.org/pdf/2012.09760v3>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

