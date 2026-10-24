# ARCHIVED — Case 7 卡名对齐（语料三 PDF ↔ 定位卡主干）

> **归档类型**：A3 迁移的回滚证据
> **归档日期**：2026-10-03
> **触发**：`verify_kb.sh` Case 7 失败 14（生物力学线 7 + 章隆彬线 7）
> **范围**：仅 `01_参考论文/生物力学线/`。
> `09_章隆彬_合作团队/` 的 7 条 PDF 缺 `_全文翻译.md`，属**另一工作线的既有缺口，不在本次归档范围**；
> 本次未处理，因此自检 Case 7 仍为 **FAIL（失败 7 / 检查 216）**，且 7 条失败全部落在该目录。

---

## 1. 原路径 → 目标路径全表

改名 6 份（全部为**同目录内**改名，无跨目录移动）：

| # | 原文件名 | 新文件名 | 目录 | 配对依据 | SHA-256 前 16 |
|---|---|---|---|---|---|
| 1 | `An Extension to the ISB Wrist Joint Coordinate System.pdf` | `AN EXTENSION TO THE ISB WRIST JOINT COORDINATE SYSTEM_ AN IN.pdf` | `B1_视觉运动学_估计与验证方法学/` | `pdfinfo-doi` | `2231712ab90557be` |
| 2 | `Comparison and Reliability of Three Goniometric Methods for Measuri.pdf` | `Comparison and Reliability of Three Goniometric Methods for.pdf` | `B1_视觉运动学_估计与验证方法学/` | `title-prefix` | `ca4bf699442c30c7` |
| 3 | `Mechanism of Limitation of Pronation-Supination of the Forearm.pdf` | `Mechanism of limitation of pronation_supination of the forea.pdf` | `B2_肌肉骨骼动力学_与可微仿真/` | `title-prefix` | `31e158a02c9bca96` |
| 4 | `Exoskeleton for Forearm Pronation and Supination Rehabilitation.pdf` | `Exoskeleton for forearm pronation and supination rehabilitat.pdf` | `B3_上肢康复_代偿与临床评估/` | `title-prefix` | `5116ad6bd42256a4` |
| 5 | `Limited Forearm Motion Compensated by Thoracohumeral Kinematics.pdf` | `Limited Forearm Motion Compensated by Thoracohumeral Kinemat.pdf` | `B3_上肢康复_代偿与临床评估/` | `title-prefix` | `95267fb8fee3e98b` |
| 6 | `Continuous estimation of finger joint angles using muscle activation inp.pdf` | `Continuous estimation of finger joint angles using muscle ac.pdf` | `B5_视觉运动学到动力学_桥接/` | `title-prefix` | `28d5fea37193c8e0` |

回收 1 份（**按机制 X1 走回收站，不进归档**）：

| 原路径 | SHA-256 前 16 | 理由 |
|---|---|---|
| `01_参考论文/生物力学线/B1_视觉运动学_估计与验证方法学/Exoskeleton for forearm pronation and supination rehabilitat.pdf` | `d6c826fca33cc8b6` | 与 B3 同篇同内容重复副本（Andreasen 2004, IEEE EMBS 26th）。字节数 205,341 == 205,341、`pdftotext` 输出 16,351 字符 == 16,351 字符、diff 仅 16 行空白差异。**卡与正本均在 B3**，B1 这份无孪生卡。 |

机器可读全表：[`rename_manifest.json`](rename_manifest.json)（SHA-256 `6df962183bc64d99ce127aa610670fa490c523421107c00d5ae6f05b11b0768b`）

---

## 2. 归档原因（被什么取代）

**新命名规则取代了人工命名。** 权威规则是 `build_biomech.py` 的
`safe_name(title_of(p), NAME_MAXLEN=60)` —— 卡名与 PDF 名**必须同一切片长度**（`download_oa.py` 用同一常量）。
上述 7 份 PDF 是从出版社/arXiv 直接下载所得，文件名是**来源站点的截断标题**（长度 60–70 不一、大小写与标点各异），
与卡主干不一致，导致 Case 7 判为「缺对应 `_Abstract_Card.md` 孪生」。

📌 方向是**以卡为准**，不是反过来：卡由脚本按固定规则生成、可复现；PDF 名是人工的。
把可复现物去对齐不可复现物才是本末倒置。

**遗留问题（未随本次归档闭合）**：`An Extension to the ISB Wrist Joint Coordinate System` 一篇的
`canonical_title` 是全大写（Crossref 原样返回），而 `title` 是正常大小写。
`build_biomech.py:175` 有意让 `title_of()` 优先取 `canonical_title`，
因此该卡与 PDF 的**文件名是全大写**。这是设计行为而非缺陷，但可读性差。
改 `title_of()` 的优先级会牵动全部 111 个文件名，**不在本次范围**，留待专项。

---

## 3. 仍被引用？

**否。** 6 份 PDF 与 1 份重复副本均未被任何 `.md` 文档以**文件名**引用，故本归档不违反机制 X4。

核验方法（⚠️ 方法本身有个坑，第一次查错了）：直接 `grep -F "<旧名>"` 会**误命中** ——
`Comparison and Reliability of Three Goniometric Methods for Measuring Forearm Supination and Pronation`
这条**完整题名**恰好以旧截断名 `Comparison and Reliability of Three Goniometric Methods for Measuri` 开头，
在 `06_附录/需你下载的论文清单.md`、`缺口台账.md`、`待下载论文总清单.md`、两张 `LIST.md` 与定位卡里各命中一次，
共 7 处 —— 但**全是题名文本，没有一处是文件名引用**。

正确的核验是搜**链接形式**：

```
grep -rnoE "\[[^]]{0,80}\]\([^)]*\.pdf\)|[A-Za-z0-9 _%()-]{10,}\.pdf" --include="*.md" . \
  | grep -v "^./11_归档/" | grep -iE "<旧名>"
```

结果为**空**。现存指向这些论文的链接全部落在 `../01_参考论文/生物力学线/B?_*/<card>_Abstract_Card.md` 上，
即指向**卡**（未改名），不是 PDF。

---

## 4. 可逆性

| 环节 | 可逆方式 |
|---|---|
| 6 份改名 | **完全可逆**。按 `rename_manifest.json` 的 `to → from` 反向 `os.rename`；每条都带原 SHA-256，改完可逐条校验字节一致。 |
| 1 份回收 | **可恢复**。走 `mavis-trash`（可恢复删除），未做不可逆处理。 |
| `build_biomech.py` 的 K39 补丁 | 可用本文件 §6 记录的 diff 反向应用。 |

---

## 5. 复活条件

- 若将来 `title_of()` 的优先级被调整（见 §2 遗留问题），本清单 6 条改名**必须整体重做**，
  届时以新规则重算 `safe_name(..., NAME_MAXLEN)`，不要沿用本表。
- 若 B1 未来新增「Exoskeleton for forearm pronation and supination rehabilitation」条目，
  须从回收站取回 `d6c826fca33cc8b6` 那份，否则需重新下载。
- 任何时候怀疑本归档的配对依据有误，用 §1 表的 SHA-256 逐条比对即可，无需重新推断。

---

## 6. 同期修复：`build_biomech.py` 的 `if card.exists(): continue`（台账 K39）

本次对齐暴露出一个更上游的缺陷：定位卡是**一次性快照**，resolve 阶段对元数据的更正无法传播。

**原实现**（`build_biomech.py:302`）：

```python
card = d / ('%s_Abstract_Card.md' % safe_name(title_of(p), NAME_MAXLEN))
if card.exists():
    continue
```

**实测后果**：111 张卡中有 **25 张**与 `biomech_resolved.json` 矛盾，其中 **19 张的 `evidence_grade` 是旧值**。
即：卡写 D 级、而按同一份 resolved 生成的 `LIST.md` 索引写 B 级 —— 同一篇论文两处定级矛盾，
且查卡的人会拿到被压低的等级。题名陈旧 9 张（含 2 张全大写/正常大小写不一致）。

**修法**（内容比对代替存在性判断）：

```python
text = '\n'.join(body)
if card.exists() and card.read_text(encoding='utf-8') == text:
    continue
card.write_text(text, encoding='utf-8')
ncard += 1
```

**验证**：首次重跑重写 83 张、卡总数 111→111（**0 新增 / 0 丢失 / 0 孤儿**）；
紧接第二次重跑输出「本次新建定位卡: 0」，证明幂等。
逐字段复核 111 张卡的 `title/thread/lens/year/doi/arxiv_id/evidence_grade/confirm_sources`，
`evidence_grade` 不一致 **0 处**。

⚠️ **残留的 23 处 `title` 差异不是缺陷**：`build_biomech.py:175` 有意用 `title_of()`（`canonical_title` 优先）
填卡题名，故 13 条 arXiv 优先件的卡题名来自 `canonical_title` 而 resolved 的 `title` 为空串（K38 降级路径）。
核对脚本若拿卡题名去比 resolved 的 `title` 字段，会把这 13 条误报为「陈旧」——**换字段比对即可，不要改代码**。
