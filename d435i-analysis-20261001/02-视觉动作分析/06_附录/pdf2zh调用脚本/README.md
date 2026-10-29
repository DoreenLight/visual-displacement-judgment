# pdf2zh 批量翻译指南

本目录是「视觉→关节角 Seq2Seq 翻译」研究项目的 PDF 翻译流水线脚本。

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