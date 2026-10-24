# pdf2zh 批量翻译指南

本目录是「视觉→关节角 Seq2Seq 翻译」研究项目的 PDF 翻译流水线脚本。

> **2026-10-01 起改用本目录自建流水线**（`paper_manifest.py` + `translate_all.py` +
> `verify_translations.py`），不再走 `run_batch_cli.sh`。原因：原 `.env` 里的
> `OPENROUTER_MODEL=stealth/ox-alpha` 已从 OpenRouter 下线（接口 404），pdf2zh 会把
> 错误响应当正文写入，产出「纯英文译文」。详见 STATUS.md §7.1。
> 下方 §0–§3 保留为历史记录，§4 起为现行口径。

## 现行流水线（2026-10-01）

### 一键重跑

```bash
PY=/Users/xbpd/Projects/pdf2zh/.venv/bin/python

$PY paper_manifest.py --list          # 1. 清单校验：29 篇 PDF → 英文抽取
bash run_extract_missing.sh            # 2. 补跑缺失的 MinerU 英文抽取
$PY translate_all.py --workers 8      # 3. 全文翻译（断点续跑）
$PY translate_all.py --repair         # 4. 重跑被拒答/幻觉图污染的块
$PY translate_all.py --fixnums        # 5. 重跑数字对不上的块（治 8.57→8.47）
$PY translate_all.py --deecho         # 6. 删英文复读段 + 清残留 ```（纯本地，不调 API）
$PY verify_translations.py            # 7. 逐篇质量校验（29/29 通过则退出码 0）
```

### 脚本职责

| 脚本 | 职责 |
|---|---|
| `paper_manifest.py` | 29 篇的 PDF → 英文抽取映射清单（`--list` 可打印页数/是否已有抽取） |
| `run_extract_missing.sh` | 对缺英文抽取的篇目跑 MinerU（`--skip-translate`），产物进 `_translation_outputs/` |
| `translate_all.py` | MiniMax-M3 全文翻译。分块、断点续跑、拒答重试、去回显、幻觉图片清理 |
| `verify_translations.py` | 质量校验，退出码非 0 即有未通过项 |

### 翻译口径

| 项 | 口径 |
|---|---|
| 范围 | 全文，含 References |
| 公式 `$$...$$` | 逐字原样保留，**不进模型** |
| 代码 ``` ``` ``` | 逐字原样保留，**不进模型** |
| 表格 `<table>` | 整表原样保留（表头英文），**不进模型** |
| 图片 `![](images/…)` | 逐字原样保留，**不进模型** |
| 参考文献 | 标题译中文；作者/刊名/年份/卷期页码/DOI 保持英文 |
| 专有名词 | 首次「中文译名（English Original）」；数据集/模型/机构名保留英文 |

**为什么结构性内容不进模型**：实测模型会丢图片、丢整表、丢块公式、丢代码块。
详见 STATUS.md §7.1。分块器在 `chunk_markdown()`，可用
`$PY translate_all.py --dry-run` 验证「重建后与原文结构零差异」。

### 四个必须知道的坑

1. **拒答**：孤立碎片（单独的 `(a)`、`Table 2 Continued`、光杆小节标题）会触发模型
   拒答。`_translate_one()` 先正常翻，检测到拒答就用 `NUDGE` 提示重试，再失败原样透传。
   改 `REFUSAL` 正则时务必跑 `/tmp/test_refusal.py` 同级回归（32 例）——
   单用关键词会误伤正文里的「无法…」，必须与第二人称称呼成对匹配。
2. **英文复读**：模型会先原样贴英文再给译文。`strip_echo()` 删除，但 `_is_mathy()`
   必须排除公式/代码块，否则会把公式当复读删掉（这个坑踩过一次）。
3. **数字错位**：模型会改写数字（8.57→8.47）。`--fixnums` 检出后重译。
4. **缓存命名空间**：`translate_all.py` 里 `key = f"v3::{name}"`。改分块逻辑就
   必须升版本号，否则会命中旧结构的缓存。

### 校验器口径（`verify_translations.py`）

写校验器时踩过的坑，一并记下：

- **中文占比只能算「正文散文」**：要排除 References（作者/刊名保持英文）、`<table>`（整表保留
  英文）、`$$公式$$`、` ```代码``` `、图片引用。直接对全文算会被表格拉到 0.24。
- **不能按 `## References` 截断取数**：MinerU 会把末张大表排在参考文献之后
  （Jamsrandorj2022-EMBC 的 Table VI），一截断就把表里数据全判成丢失。
  正确做法：正文取 References 之前，再把全文里的 `<table>` 拼回去。
- **数字要按「值」比而不是按字符串比**：2.4 billion→24 亿、13.7 million→1370 万
  是正确换算，逐字比会误报。日期正则也不能写宽——`\d{1,2}\.\d{1,2}` 会把 8.57 当成日期吃掉。
- **只查「少了」不查「多了」**：模型可能把长行内式改写成行间式，块公式数变多不是缺陷。

---

## 历史记录：pdf2zh 原流水线（模型已失效，勿再使用）

## 0. 一次性环境检查

```bash
ls /Users/xbpd/Projects/pdf2zh/.venv  # 必须存在
ls /Users/xbpd/Projects/pdf2zh/.env   # 必须包含 OPENROUTER_* 三行
```

如果 `.venv` 不存在：双击 `/Users/xbpd/Projects/pdf2zh/pdf2zh.bat` 完成首次安装（约 10 分钟，下载 7 GB 模型）。

如果 `.env` 没有 `OPENROUTER_*`：复制下面三行进去：

```bash
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=stealth/ox-alpha
OPENROUTER_API_KEY=sk-or-v1-你的key
```

---

## 1. 两种批量模式

### 模式 A：Web UI 手动（推荐首次使用）

```bash
bash run_batch.sh
```

流程：
1. 自动启动 pdf2zh Web UI（后台，端口 7862）
2. 自动等 UI 就绪（最多 60 秒）
3. 自动用 `tools/test_batch_via_ui.py` 把 `01_参考论文/` 下所有 PDF 提交到 UI
4. UI 后台翻译，进度实时打到日志
5. 输出落到 `01_参考论文/_translation_outputs/<论文名>/auto/<论文名>_全文翻译.md`

### 模式 B：CLI 自动化（推荐二次使用）

```bash
bash run_batch_cli.sh
```

流程：
1. 不启 Web UI，直接对每篇 PDF 调 `pdf_to_zh_md.py`
2. 输出落到 `01_参考论文/_translation_outputs/<论文子目录>__<论文stem>/<论文stem>_全文翻译.md`

**两个模式的差异**：

| 项 | A (Web UI) | B (CLI) |
|---|---|---|
| 启动速度 | 慢（等 UI） | 快（直接 pip 调脚本） |
| 翻译速度 | 同样（都是走 mineru + OpenAI 客户端） |
| 看进度 | Gradio Web UI + 日志 | 直接打 stdout |
| 重跑 | 跳过已完成的 PDF（pdf2zh 内置断点续传） |
| Token 用量 | 相同 |

---

## 2. 输出位置与下游整合

### 2.1 原始输出（pdf2zh 自己写的位置）

| 模式 | 路径 |
|---|---|
| A | `01_参考论文/_translation_outputs/<论文名>/auto/<论文名>_全文翻译.md` |
| B | `01_参考论文/_translation_outputs/<论文子目录>__<论文stem>/<论文stem>_全文翻译.md` |

### 2.2 整理到对应子目录

跑完翻译后，执行：

```bash
bash organize_translations.sh
```

这个脚本会：
1. 扫描 `_translation_outputs/` 下所有 `_全文翻译.md`
2. 对每篇，根据论文子目录名找到对应的论文子目录
3. 把 `_全文翻译.md` 复制（不移动）到对应论文子目录下的 `_全文翻译.md`
4. `_translation_outputs/` 保留作为备份

结果：每个论文子目录里既有 `*.pdf`，也有 `<论文名>_全文翻译.md`。

---

## 3. 中断与续传

pdf2zh 自身有断点续传：
- 中途关掉 Web UI / CLI 进程 → 下次启动会跳过已完成的 PDF
- MinerU 模型下到本地 → 不会重复下

清空重跑：

```bash
rm -rf 01_参考论文/_translation_outputs/*
```

---

## 5. 切换模型（可选）

`.env` 里目前配的是：
- `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`
- `OPENROUTER_MODEL=stealth/ox-alpha`（OpenRouter 免费模型）

想换其它模型（比如付费但更准的）：
- 去 <https://openrouter.ai/models> 选
- 把 `OPENROUTER_MODEL` 改成 `anthropic/claude-3.5-sonnet` 之类
- 重启 Web UI 或重跑脚本

---

## 6. 常见问题

| 现象 | 怎么办 |
|---|---|
| `未找到 .venv` | 跑首次 pdf2zh.bat |
| `OPENROUTER_API_KEY missing` | 补 .env |
| `mineru failed` | 网络问题，挂代理重试 |
| 翻译出来是英文 | 检查 `OPENROUTER_MODEL` 是否真的指向中文模型 |
| 翻译出来乱码 | 检查 chunk_size（默认 3000），调到 1500-2000 重试 |
| 想加 PDF | 放进 `01_参考论文/` 对应子目录 → 重跑脚本 |