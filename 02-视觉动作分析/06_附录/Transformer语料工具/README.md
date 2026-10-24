# Transformer 语料工具（可复现构建链）

> **用途**：构建与自校验 `01_参考论文/运动理解/Transformer_发展线/` 语料库。
> **原则**：全程不硬编码 arXiv ID、不做付费墙绕过；每条结论都留可复算的机器记录。
> **运行位置**：`06_附录/Transformer语料工具/`（工作目录即本目录）。

---

## 1. 脚本与执行顺序

| 顺序 | 脚本 | 作用 | 产物 | 是否联网 |
|---|---|---|---|---|
| 1 | `resolve_corpus.py` | 以 `ti:` 标题检索 arXiv 官方 API，逐条做标题相似度比对（≥0.90 自动采信），冒号标题自动降级到 5 级回退检索 | `corpus_resolved.json` | 是 |
| 2 | `adjudicate.py` | 对低相似度条目，用**我给出的 ID 假设**反查 arXiv 真实标题来证实/证伪 | 回填 `corpus_resolved.json` | 是 |
| 3 | `rescue.py` | 对仍未命中的条目改用**标题片段**检索，给出可救候选（不自动采信） | `rescue_candidates.json` | 是 |
| 4 | `download_pdfs.py` | 下载 arXiv 开放获取 PDF，三重校验（`%PDF` magic + 体积下限 + `file` 判定） | PDF 文件 + `corpus_download_report.json` | 是 |
| 5 | `gen_cards.py` | 生成**摘要级定位卡**（显式自述「不是全文翻译」） | `*_Abstract_Card.md` ×N | 否 |
| 6 | `build_index.py` | 生成可检索索引、语料库 README、7 份线程 LIST.md、**取件入口指针** | `index.json` / `README.md` / `*/LIST.md` / `取件入口.md` | 否 |
| 7 | `build_page.py` | 生成双轴矩阵 + 时间线 + 可检索清单的可视化页 | `08_三维动作分析展示台/Transformer_发展线图谱.html` | 否 |
| 8 | `verify_citations.py` | **反查推演文档里每一个手写 arXiv ID** 是否真实、是否与语料一致 | 控制台报告 | 是 |

## 2. 一键复现

```bash
cd 06_附录/Transformer语料工具
python3 resolve_corpus.py        # 增量；--force 全量重查；--only 3,7 强制重查指定序号
python3 adjudicate.py
python3 download_pdfs.py         # --retry-failed 重试失败项
python3 gen_cards.py
python3 build_index.py
python3 build_page.py
python3 verify_citations.py
bash ../自检脚本/verify_kb.sh
```

## 3. 输入文件

| 文件 | 说明 | 维护方式 |
|---|---|---|
| `corpus_candidates.json` | **人工编制的候选清单**（理论搜索空间）。含每条的阶段/线程/迁移价值/note/DOI | 手工编辑；补论文从这里加 |
| `corpus_resolved.json` | arXiv 官方 API 解析结果 | 由脚本生成 |
| `corpus_download_report.json` | 下载结果（成功/失败/失败原因） | 由脚本生成 |
| `rescue_candidates.json` | 抢救检索的可救候选，**需人工确认后才写入语料** | 人工裁定 |

## 4. 关键设计决策（为什么这么做）

| 决策 | 理由 |
|---|---|
| **以标题检索代替硬编码 ID** | 记忆中的 ID 会错，且错了极难自查。实测 17 处错误中有多处 ID 指向天文学/数学/气象学等完全无关论文 |
| **标题相似度阈值 0.90 自动采信，0.74–0.90 转人工复核** | 自动化与人工之间留缓冲带；低于 0.74 一律不采信 |
| **冒号标题 5 级回退检索** | 实测 arXiv `ti:"..."` 遇冒号整条落空（PoseFormer / METRO / Performer / LLaVA 等均受影响） |
| **缓存键用标题而非序号** | 补论文或改标题时不会错位复用旧结果 |
| **下载三重校验** | 防止把 HTML 错误页当 PDF 存入库 |
| **定位卡显式写「不是全文翻译」** | 摘要级产物若被当成全文翻译引用，就是知识库最典型的失真；由 `verify_kb.sh` Case 18 强制 |
| **`verify_citations.py` 独立于建库流程** | 建库正确 ≠ 论证正确。推演文档里的引用必须单独反查——本次即靠它拦下 5 处错误引用 |

## 5. 已知限制（如实声明）

- **arXiv 的 `ti:` 检索不可靠**：多个变体 + `id_list` 兜底后，仍有 25 条未命中。其中多数是**期刊论文或会议论文未上 arXiv**（如 TNSRE / MDPI / Frontiers / IEEE），**不是工具缺陷**，须经学校数据库人工取件。
- **T2M / TATS 未核验**：这两篇是「时间感知 token 粒度」论证的关键引用，但多次检索未命中。**在其原文取到之前，相关表述不得写入论文结论。**
- **迁移价值/阶段/线程为人工判断**，不是 arXiv 客观字段；卡片中已与官方元数据分列标注。
- 下载仅限 arXiv 开放获取版本，**不做任何付费墙绕过**。

---

---

## 生物力学流程已独立

语料三现行入口为[生物力学语料工具](../生物力学语料工具/README.md)。本目录仅维护 TF 流程、TF 翻译及其取件工具；原 BM 备份和历史日志仍按原路径留证，不是现行入口。
