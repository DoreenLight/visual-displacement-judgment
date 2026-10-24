# RGB 直接竞品：2026 Chao et al. (Measurement)

**这是当前对你原始方案最大的 novelty 冲击**。必须深度阅读并明确写出"我们和它有什么不同"。

---

## 1. Chao et al. — Measurement 2026 ⭐⭐⭐⭐⭐ 必读

| 字段 | 值 |
|---|---|
| 标题 | Lower limb joint angle estimation using a single RGB camera |
| 期刊 | Measurement (Elsevier) |
| 年份 | 2026 |
| DOI | `10.1016/j.measurement.2026.120772` |
| 期刊页 | <https://doi.org/10.1016/j.measurement.2026.120772>（Elsevier） |
| **PDF 状态** | ✅ **已获取原文**（`Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计.pdf`，2,952,500 B / **10 页**，Elsevier 官方版） |
| **中文精读** | ✅ `Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计_全文翻译.md`（**基于原文的全译级精读 v2**，`[未验证]` 已清零） |
| **⚠️ 本目录论文数** | **2 篇**：Chao 2026 + 同团队前身 Jamsrandorj 2023 EMBC（均已有 PDF + 翻译） |
| 归档留证 | 📦 `Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计_reconstruction.md`（原深度替代包，现为取件记录 + 竞品对照） |
| arXiv 备查 | ❌ **已穷尽渠道，确认无 preprint**（arXiv 全文检索该标题 + 全部 6 位作者姓名组合均无命中；OpenAlex `locations` 仅含 `publishedVersion`，无任何 repository/submittedVersion 版本；该团队同期 2021/2022 EMBC 工作也均无 arXiv 版，无自存预印本习惯）。详见 `Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计_reconstruction.md` §6.1 |
| 优先级 | ⭐⭐⭐⭐⭐ 必读 |
| 与你的方向 | **直接竞品**：单目 RGB → CNN → Transformer → 髋/膝/踝角度，end-to-end |

**方法骨架**：
```
RGB video
  → CNN（空间特征）
  → Transformer（时间 gait dynamics）
  → hip/knee/ankle angles（端到端角度回归，非角度）
```

**关键证据**：
- 不同视角、速度、异常步态下 ICC > 0.92
- 损失函数 `L = L_angle + λ L_angular_velocity`
- 端到端，不走 2D pose → 3D pose → IK 的两阶段 pipeline

**我们必须回答的 3 个问题**：
1. 我们和它到底有什么方法上的区别？（不只是「换 token 化」这种命名差异）
2. 我们的 anatomy-aware tokenizer 在什么指标上能跑赢？
3. 我们能不能在它的实验数据集上跑出对照？（最好直接做对比实验）