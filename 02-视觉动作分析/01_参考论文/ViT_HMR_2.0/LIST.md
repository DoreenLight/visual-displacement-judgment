# ViT → 人体姿态：HMR 2.0 (Humans in 4D)

视觉 Transformer 直接出人体姿态参数的代表工作。**视觉 token → pose 参数**这条路已经成熟，我们要做的是把目标参数从 SMPL 改成临床关节角。

---

## 1. HMR 2.0 — ICCV 2023 ⭐⭐⭐⭐⭐ 必读

| 字段 | 值 |
|---|---|
| 标题 | Humans in 4D: Reconstructing and Tracking Humans with Transformers |
| 作者 | Shubham Goel, Georgios Pavlakos, Jathushan Rajasegaran, Angjoo Kanazawa, Kostas Daniilidis |
| 会议 | International Conference on Computer Vision (ICCV) |
| 年份 | 2023 |
| arXiv | `2305.20091` |
| arXiv 链接 | <https://arxiv.org/abs/2305.20091> |
| 优先级 | ⭐⭐⭐⭐⭐ 必读 |
| 与你的方向 | **核心方法学参照**：ViT image encoder + SMPL query cross-attention |

**架构（你必须画出来）**：
```
Image
  → ViT（image tokens）
  → Transformer Decoder + SMPL query token（cross-attention）
  → (θ, β, π)  ← SMPL pose + shape + camera
```

**为什么对我们尤其重要**：
- 已经是 image tokens → pose parameters 的存在方式
- **和我们的区别**：目标表征不同
  - 它：SMPL pose parameters（人体网格参数）
  - 我们：临床关节角（髋/膝/踝屈伸、收展、旋转等自由度）
- 还要把单帧升级为时序，并加 biomechanical grammar

**读完后回答**：
1. ViT backbone 是哪一档（ViT-B/L/H）？
2. SMPL query 是怎么初始化的？
3. 训练用了什么数据集？
4. 能否借鉴它的训练策略到我们的 kinematic grounding task？