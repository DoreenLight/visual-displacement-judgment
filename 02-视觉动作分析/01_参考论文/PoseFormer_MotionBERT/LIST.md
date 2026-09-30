# PoseFormer / MotionBERT 系列：2D Pose → 3D Pose

这条路是「姿态骨架翻译」的存在方式，**严格说不直接出关节角**，但思路完全可借鉴。

---

## 1. PoseFormer — ICCV 2021 ⭐⭐⭐⭐

| 字段 | 值 |
|---|---|
| 标题 | 3D Human Pose Estimation with Spatial and Temporal Transformers (PoseFormer) |
| 作者 | Ce Zheng, Sijie Zhu, Matias Mendieta, Taojiannan Yang, Chen Chen, Zhengming Ding |
| 会议 | ICCV |
| 年份 | 2021 |
| arXiv | `2103.10455` |
| arXiv 链接 | <https://arxiv.org/abs/2103.10455> |
| 与你的方向 | 2D joints → Spatial-Temporal Transformer → 3D joints |

**架构关键**：
- Spatial Transformer：同帧关节间关系
- Temporal Transformer：跨帧时间关系

---

## 2. PoseFormerV2 — CVPR 2023 ⭐⭐⭐⭐

| 字段 | 值 |
|---|---|
| 标题 | PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation |
| 会议 | CVPR |
| 年份 | 2023 |
| arXiv | `2303.17472` |
| arXiv 链接 | <https://arxiv.org/abs/2303.17472> |
| 与你的方向 | 加入频域表示，提高长序列建模和对 2D 检测噪声的鲁棒性 |

---

## 3. MotionBERT — ICCV 2023 ⭐⭐⭐⭐⭐ 必读

| 字段 | 值 |
|---|---|
| 标题 | MotionBERT: A Unified Perspective on Learning Human Motion Representations |
| 作者 | Wentao Zhu, Xiaoxuan Ma, Zhaoyang Liu, Libin Liu, Wayne Wu, Yizhou Wang (PKU + Shanghai AI Lab) |
| 会议 | ICCV |
| 年份 | 2023 |
| arXiv | `2210.06551` |
| arXiv 链接 | <https://arxiv.org/abs/2210.06551> |
| 与你的方向 | Dual-stream Spatio-temporal Transformer，从带噪/缺失的 2D skeleton 恢复 3D motion；学到的表征包含 geometric / kinematic / physical 知识 |

**为什么必读**：
- 它明确说出学到的表征包含 kinematic 知识 ← **和我们的 kinematic language 思路完全对齐**
- 给我们一个强证据：Transformer 学 motion 是学得出 kinematic 结构的

---

## 4. MotionAGFormer — WACV 2024 ⭐⭐⭐

| 字段 | 值 |
|---|---|
| 标题 | MotionAGFormer: Enhancing 3D Human Pose Estimation with a Transformer-GCNFormer Network |
| 作者 | Soroush Mehraban, Vida Adeli, Babak Taati (University of Toronto) |
| 会议 | WACV |
| 年份 | 2024 |
| arXiv | `2310.16288` |
| arXiv 链接 | <https://arxiv.org/abs/2310.16288> |
| 与你的方向 | Transformer + GCN 混合：global relation + local joint relation |

**对我们的意义**：
- 可以借鉴它的混合架构思路
- 我们是否也需要 GCN 来建模 kinematic chain？