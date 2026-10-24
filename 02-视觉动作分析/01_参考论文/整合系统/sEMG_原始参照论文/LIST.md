# sEMG 原始参照论文

这是我们整个项目的「种子论文」——你最初在 IEEE Xplore 看到的那篇。它建立了「神经控制信号 → 关节角度 = 语言翻译」的范式。我们要做的是把「神经控制信号 (sEMG)」换成「视觉表征」，但保留并升级翻译范式。

**重要**：本目录同时含 2 篇 **Wei Zijun 团队真实下载的 PDF**（包括用户原目标团队的 TNSRE 2024 综述论文），可作为种子论文的真正同团队参照物。

---

## 0. Wei, Zhang, Xie — TNSRE 2024（已下载 PDF, 同团队真综述）⭐⭐⭐⭐⭐

| 字段 | 值 |
|---|---|
| 标题 | Continuous Motion Intention Prediction Using sEMG for Upper-Limb Rehabilitation: A Systematic Review of Model-Based and Model-Free Approaches |
| 作者 | Zijun Wei, Zhi-Qiang Zhang, Sheng Quan Xie（**完全同 JBHI 2026 团队**）|
| 期刊 | IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE) |
| 年份 | 2024 |
| DOI | `10.1109/TNSRE.2024.3383857` |
| IEEE Xplore | <https://ieeexplore.ieee.org/document/10487972/> |
| **PDF 状态** | ✅ **已下载 + 已翻译** (940 KB, 19 页) |
| OpenAlex 标记 | **OA=True**（cc-by 许可）|
| 与你的方向 | **同团队、同主题、同 sEMG 信号的综述论文**——能告诉你在"连续运动意图预测"这个大领域里，模型基 vs 无模型基方法的全景 |

**为什么是 gold mine**：
- 同一作者团队（Wei, Zhang, Xie）和 JBHI 2026 完全相同
- 是 **综述论文**——会引用 2026 之前所有 sEMG → motion 的工作
- TNSRE 期刊，2024 发表，正好在用户原目标论文 2026 之前
- 包含对 muscle-anatomy-informed、deep learning、transformer 等方法的覆盖
- 拿到这篇，等于拿到一个「通向 JBHI 2026 论文的桥」

**与 JBHI 2026 目标论文的关系**：
| 维度 | TNSRE 2024 综述 | JBHI 2026 目标 |
|---|---|---|
| 作者 | Wei, Zhang, Xie | Wei, Zhang, Xie |
| 主题 | sEMG → 上肢运动意图（综述）| sEMG → 关节角度（Seq2Seq） |
| 输出 | 方法分类（model-based / model-free）| 连续关节角 |
| 用途 | 帮你理解整个 sEMG motion 领域的全貌 | 你的种子论文 |
| 状态 | ✅ 已下载可读 + 已译 | ✅ **已获取原文 + 全译级精读** |

---

## 1. Wei, Zhang, Xie — JBHI 2025/2026 ⭐⭐⭐⭐⭐ 必读

| 字段 | 值 |
|---|---|
| 标题 | A Transformer Framework Informed by Muscle Anatomy and Sequence-to-Sequence Translation for Continuous Joint Kinematics Prediction Using sEMG |
| 作者 | Zijun Wei, Zhiqiang Zhang, Sheng Quan Xie |
| 会议/期刊 | IEEE Journal of Biomedical and Health Informatics (JBHI) |
| 卷期 | Vol. 30, No. 2, 2026 |
| DOI | `10.1109/JBHI.2025.3589889` |
| IEEE Xplore | <https://ieeexplore.ieee.org/document/11886441/> |
| arXiv 备查 | （无）|
| **PDF 状态** | ✅ **已获取原文**（`Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测/Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测.pdf`，4,273,786 B / **13 页**，IEEE 官方版） |
| **中文精读** | ✅ `Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测/Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测_全文翻译.md`（**基于原文的全译级精读**，`[未验证]` 已清零） |
| 优先级 | ⭐⭐⭐⭐⭐ 必读 |
| 与你的方向 | **源项目**，参考其 muscle-anatomy-informed tokenization、Seq2Seq 翻译范式、300 ms 提前预测思路 |

**为什么必读**：
- 论文命名直接出现 "Sequence-to-Sequence Translation"
- 把 sEMG 当作 source language、关节角当作 target language
- 用肌肉解剖结构指导 3D CNN 特征
- 报告了约 300 ms 的运动学提前预测

**读完后回答 4 个问题**：
1. 它怎么定义 sentence/word 的？
2. muscle-anatomy-informed 是怎么注入到 3D CNN 的？
3. 300 ms 提前预测是怎么评估的？
4. 我们能直接复用什么它？