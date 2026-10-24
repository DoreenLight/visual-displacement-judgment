# OpenCap + 传统视觉生物力学 pipeline

**临床/工程基线**。注意：绝大多数 biomechanical markerless motion capture 走的是「2D detection → triangulation → IK」，不是端到端。我们要做的是端到端 + 结构化 translation。

---

## 1. OpenCap — PNAS / Scientific Data 2023 ⭐⭐⭐

| 字段 | 值 |
|---|---|
| 标题 | OpenCap: Democratizing motion capture through smartphone videos |
| 年份 | 2023 |
| 期刊 | **PLOS Computational Biology**（本地文件 `OpenCap：从智能手机视频获取人体运动动力学/OpenCap：从智能手机视频获取人体运动动力学.pdf`，已由文件名与正文确认） |
| 链接 | <https://www.opencap.ai/> |
| arXiv 备查 | （OpenCap 有自己的 white paper，可在官网下载） |

**方法骨架**：
```
手机视频
  → 计算机视觉（HPE + 三角测量）
  → OpenSim 肌骨模型
  → 关节角度 + 关节力
```

**关键证据**：
- 原始验证：关节旋转 MAE ≈ 4.5°
- 2024 ACL 康复：MAE ≈ 3.85°、RMSE ≈ 4.34°

**对我们的意义**：
- **临床/工程基线**：我们必须跑过 OpenCap 的数字才能声明「端到端先进」
- OpenCap 走的是 2D → 3D → IK 三阶段，而我们是端到端

---

## 2. 2026 Markerless Motion Capture 综述 ⭐⭐⭐

| 字段 | 值 |
|---|---|
| 标题 | Markerless motion capture methodologies in biomechanics: From 2D images to 3D joint angles |
| 年份 | 2026 |
| 类型 | Review |

**对我们的意义**：
- 综述明确说绝大多数 biomechanical markerless motion capture 仍走 2D → IK
- 给我们留下「端到端视觉 → 临床关节角」的空间
- 必须引用进 Related Work

**下载注意**：
- 综述可能在期刊官网，需要机构访问
- 备查：可考虑在 ResearchGate / Google Scholar 找作者主页版本