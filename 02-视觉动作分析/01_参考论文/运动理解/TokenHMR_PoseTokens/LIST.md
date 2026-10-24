# TokenHMR：pose tokenization 是「语言翻译」最接近的存在方式

**这篇是「翻译」二字最名副其实的存在工作**：把人体姿态变成离散 token，限制预测空间。

---

## 1. TokenHMR — CVPR 2024 ⭐⭐⭐⭐⭐ 必读

| 字段 | 值 |
|---|---|
| 标题 | TokenHMR: Advancing Human Mesh Recovery with a Tokenized Pose Representation |
| 会议 | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) |
| 年份 | 2024 |
| arXiv | `2404.16752` |
| arXiv 链接 | <https://arxiv.org/abs/2404.16752> |
| 优先级 | ⭐⭐⭐⭐⭐ 必读 |
| 与你的方向 | **概念最接近**：pose vocabulary + token prediction，正是「source language → target language」的实例化 |

**核心思想（必须背下来）**：
- encoder 把连续 pose 映射到 discrete pose tokens
- 通过 pose vocabulary 限制预测到合理姿态空间
- 把 HMR 重定义为 token prediction

**和 TokenHMR 的差异化（必须想清楚）**：
- 它是 image → SMPL pose tokens（单帧人体网格）
- 我们是 video → clinical joint-angle tokens（时序临床自由度）
- 我们要加 biomechanical grammar（ROM、velocity、acceleration、FK 等约束）

**我们要不要复用它的 pose VQ tokenizer**：
- 是 ✓（作为 inspiration）
- 但我们要重新设计 kinematic vocabulary（针对关节角离散化）