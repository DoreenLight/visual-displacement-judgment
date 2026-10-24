# METRO：图像 → 3D 关节点 + 网格

视觉 Transformer 直接出人体几何的经典代表。

---

## 1. METRO — CVPR 2021 ⭐⭐⭐

| 字段 | 值 |
|---|---|
| 标题 | End-to-End Human Pose and Mesh Reconstruction with Transformers (METRO) |
| 作者 | Kevin Lin, Lijuan Wang, Zicheng Liu (Microsoft) |
| 会议 | CVPR |
| 年份 | 2021 |
| arXiv | `2012.09760` |
| arXiv 链接 | <https://arxiv.org/abs/2012.09760> |
| 与你的方向 | 图像 → 3D joints + mesh 的早期工作（end-to-end transformer） |

**对我们的意义**：
- Transformer 直接出人体几何参数
- 但它是单帧图像 → 网格，不是时序视频 → 关节角
- 我们可以参考它的 query token 初始化策略