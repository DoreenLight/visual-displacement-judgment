# 语料二（Transformer 发展线）翻译流水线 —— 隔离版

> **为什么单独一套**：本项目的主语料翻译由 `06_附录/pdf2zh调用脚本/` 承担，且**他人正在使用**
> （`translate_all.py --workers 8 --fixnums` 在跑 29 篇的数字保真修复）。
> 本流水线只服务 `01_参考论文/运动理解/Transformer_发展线/` 的 85 篇，**不读写主语料的任何文件**。

---

## 1. 隔离边界（逐条可核）

| 共享资源 | 主语料用 | 语料二用（本流水线） | 是否冲突 |
|---|---|---|---|
| `translate_all.py` | 读写 | **只 import，不改** | 否 |
| `paper_manifest.py` | 读写 | **不用**，改用 `tf_manifest.py` | 否 |
| `verify_translations.py` | 读写 | **只 import，不改** | 否 |
| 抽取产物目录 | `01_参考论文/_translation_outputs/` | `01_参考论文/运动理解/Transformer_发展线/_translation_outputs/` | 否 |
| 翻译块缓存 | `/tmp/zh_translate_cache` | `/tmp/zh_translate_cache_tf` | 否 |
| 运行报告 | `/tmp/translate_report.json`、`/tmp/verify_report.json` | `/tmp/tf_translate_report.json`、`/tmp/tf_verify_report.json` | 否 |
| 抽取 todo/日志 | `/tmp/mineru_*.{tsv,log}` | `/tmp/tf_mineru_*.{tsv,log}` | 否 |
| 翻译 API | MiniMax-M3（`api.minimaxi.com`） | 同一模型/同一凭据，但请求独立 | 否 |
| MinerU 抽取 | 云端 API（`--ocr api`） | 同一云 API，**串行单进程** | 否（不抢核） |

**关键实现**：不复制 28 KB 的 `translate_all.py`，而是用 `importlib` 载入后只替换两个模块级变量
（`CACHE_DIR`、`_manifest`）。因为 `find_new_extract()` 读的是 `_manifest.NEW_EXTRACT_DIR`，
替换 manifest 即可把抽取检索一并重定向到语料二目录。这样上游修 bug 时本流水线自动受益，且**不会分叉漂移**。

**为什么不调用 `ta.main()` / `ver.main()`**：二者都会写共享的 `/tmp/translate_report.json` 与
`/tmp/verify_report.json`，会覆盖他人正在读的报告。故本目录用自己的驱动循环，写自己的报告。

## 2. 三个脚本

| 脚本 | 作用 | 关键参数 |
|---|---|---|
| `tf_manifest.py` | 从 `01_参考论文/运动理解/Transformer_发展线/index.json` 生成 85 篇清单（**不手写条目**） | `--list` |
| `tf_extract.sh` | MinerU 云端抽取英文 md + images，串行 | — |
| `tf_translate.py` | MiniMax-M3 全文翻译（公式/代码/表格/图片不进模型，原样拼回） | `--dry-run` `--workers` `--only` `--thread` `--limit` `--repair` `--deecho` `--fixnums` |
| `tf_verify.py` | 逐篇质量校验（中文占比/章节/References/公式/表格/代码/图片落盘/拒答套话） | `--only` `--thread` |

## 3. 执行顺序

```bash
cd 06_附录/Transformer语料工具/翻译
bash  tf_extract.sh                       # 1) 抽取英文（可反复跑，已抽取的自动跳过）
/Users/xbpd/Projects/pdf2zh/.venv/bin/python tf_translate.py --dry-run   # 2) 看计划
/Users/xbpd/Projects/pdf2zh/.venv/bin/python tf_translate.py --workers 6  # 3) 翻译（可断点续跑）
/Users/xbpd/Projects/pdf2zh/.venv/bin/python tf_verify.py                # 4) 质量校验
```

翻译**断点续跑**：每块结果写 JSON checkpoint（`/tmp/zh_translate_cache_tf/`），中断后重跑只补未译块。
追加 `--fixnums` 重跑数字对不上的块（治 8.57→8.47 这类数位错位）；`--deecho` 纯本地删英文复读段，不调 API。

## 4. 翻译口径（继承主语料 `translate_all.py`，未改动）

| 项 | 口径 |
|---|---|
| 范围 | 全文，含 References |
| 后端 | MiniMax-M3（`https://api.minimaxi.com/v1`），凭据复用 `pdf_to_zh_md.load_api_key()` |
| 公式 | `$$...$$` 与 `$...$` 逐字原样保留，不进模型 |
| 代码 | ` ``` ` 块原样保留，不进模型 |
| 表格 | `<table>` 整表原样保留（表头英文），不进模型 |
| 图片 | 独占一行的图片引用原样保留，不进模型 |
| 参考文献 | 条目标题译中文；作者/刊名/年份/卷期页码/DOI 保持英文 |
| 专有名词 | 首次出现「中文译名（English Original）」；数据集/模型/机构名保留英文 |

**为什么公式/表格/图片不进模型**：主语料实测四种丢法都真实发生过（成串图片被丢、整表被吃、
块公式漏掉、代码块被丢）。改为原样拼回后结构零丢失。

## 5. 产物落位

```
01_参考论文/运动理解/Transformer_发展线/<线程>/
├── <年>_<首作者>_<题名片段>.pdf                     原论文
├── <年>_<首作者>_<题名片段>_Abstract_Card.md        摘要级定位卡
├── <年>_<首作者>_<题名片段>_全文翻译.md             ← 本流水线产物
└── images/                                          图片（从抽取目录同步）
```

定位卡与全文翻译**并存且不互相冒充**：卡片继续声明「不是全文翻译」，
`verify_kb.sh` Case 18 只校验卡片自述，Case 7 校验 PDF↔工件配对。

## 6. 已知限制

- **CPU-only 机器**（无 CUDA）。抽取走云端 MinerU API，故 CPU 影响小；但若改用本地 MinerU 会很慢。
- 85 篇总页数约 1200+ 页，翻译耗时长（经验：每块 3000 字符、并发 6）。**断点续跑，不要中途改参数**。
- 翻译**不能替代人工核读**：`tf_verify.py` 只做结构性质量核对，不判断技术内容是否译对。
- 若他人修改了 `translate_all.py` 的 `process_paper()` 签名，本驱动的调用可能需要同步调整。

## 7. 2026-10-01 阻塞：MinerU 抽取通道不可用

共享工具 `/Users/xbpd/Projects/pdf2zh/pdf_to_zh_md.py` 的 `--ocr` 选项已从 `{api}` 改为
`{precision, flash, local}`，**本目录的 `tf_extract.sh` 已同步修复**（按可用性自动选通道）。

但三条通道当前都走不通：

| 通道 | 状态 | 原因 |
|---|---|---|
| `precision` | 不可用 | 需 `MINERU_API_KEY`，本机 `.env` 未配置 |
| `flash` | 不可用 | 2026-10-01 实测 8 篇全部失败：多数 `HTTP 413`，个别 `HTTP 400 field "file_name" is not set`。2.2MB 的 UDE 也报 413，**不是体积限制，是服务端通道异常** |
| `local` | 不实用 | 本机 CPU-only、无 CUDA，本地 MinerU 极慢 |

**影响范围**：

- 语料二已抽取的 85 篇**不受影响**，`tf_run_all.sh` 正在继续翻译这批。
- 新入库的 **UDE、GPT-2** 暂时拿不到全文译文（PDF 与摘要级定位卡均已就位，不影响检索与引用）。
- 同一问题会影响 `06_附录/pdf2zh调用脚本/run_extract_missing.sh` —— 它仍在传已失效的
  `--ocr api`，argparse 会直接报 `invalid choice: 'api'`。该文件属并发工作流，未擅自修改。

**解除条件**（任一即可）：

1. 配置 MinerU Token 后 `export MINERU_API_KEY=...`，重跑 `bash tf_extract.sh`；
2. 或由共享工具维护方修复 `flash` 通道的 `file_name` 参数与 413 限制。
