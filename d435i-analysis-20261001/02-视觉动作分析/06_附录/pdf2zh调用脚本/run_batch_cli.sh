#!/usr/bin/env bash
# 批量翻译 01_参考论文/ 下所有 PDF 为中文 Markdown（CLI 模式，更快）
#
# 与 run_batch.sh 的区别：
#   - 不启 Web UI，直接对每篇 PDF 调 pdf_to_zh_md.py
#   - 不走 Gradio batch API，直接走 mineru + OpenAI 客户端
#   - 没有 Web UI 等待时间，启动更快
#   - 进度由 pdf_to_zh_md.py 自身打印
#
# 用法：
#   bash run_batch_cli.sh
#
# 注意：
#   - pdf_to_zh_md.py 在首次跑 mineru 时会下载模型（约 7 GB），慢但只一次
#   - OPENROUTER_BASE_URL / OPENROUTER_MODEL / OPENROUTER_API_KEY 从 pdf2zh/.env 读
#   - 每篇 PDF 一个子目录：_translation_outputs/<论文名>/<论文名>_zh.md

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PDF2ZH_ROOT="/Users/xbpd/Projects/pdf2zh"
PAPERS_DIR="${PROJECT_ROOT}/01_参考论文"
OUTPUT_DIR="${PAPERS_DIR}/_translation_outputs"

echo "== pdf2zh CLI 批量翻译 =="
echo "项目根：${PROJECT_ROOT}"
echo "PDF2ZH：${PDF2ZH_ROOT}"
echo "论文目录：${PAPERS_DIR}"
echo "输出目录：${OUTPUT_DIR}"
echo ""

# ---------- 环境检查 ----------
cd "${PDF2ZH_ROOT}"
if [ ! -d ".venv" ]; then
  echo "[ERROR] 未找到 .venv，请先双击 pdf2zh.bat 完成首次安装" >&2
  exit 1
fi

source .venv/bin/activate

# ---------- 加载 OPENROUTER 配置 ----------
set -a
grep -E '^OPENROUTER_(BASE_URL|MODEL|API_KEY)=' "${PDF2ZH_ROOT}/.env" > ./.env.openrouter
source ./.env.openrouter
rm -f ./.env.openrouter
set +a

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "[ERROR] .env 里没有 OPENROUTER_API_KEY" >&2
  exit 1
fi

echo "[*] 模型：${OPENROUTER_MODEL}"
echo "[*] 端点：${OPENROUTER_BASE_URL}"
echo ""

# ---------- 找所有 PDF ----------
mapfile -t PDFS < <(find "${PAPERS_DIR}" -maxdepth 3 -name "*.pdf" -type f | sort)
TOTAL=${#PDFS[@]}

if [ "${TOTAL}" -eq 0 ]; then
  echo "[ERROR] 没在 ${PAPERS_DIR} 找到任何 PDF" >&2
  exit 1
fi

echo "[*] 共 ${TOTAL} 篇 PDF 待翻译"
mkdir -p "${OUTPUT_DIR}"

# ---------- 逐篇调用 ----------
SUCCESS_LIST=()
FAIL_LIST=()
COUNT=0

for pdf in "${PDFS[@]}"; do
  COUNT=$((COUNT + 1))
  stem="$(basename "${pdf}" .pdf)"
  # 子目录：用 paper 子目录名 + 文件名 stem，避免同名冲突
  paper_subdir="$(echo "${pdf}" | sed -E "s|${PAPERS_DIR}/([^/]+)/.*|\1|")"
  pdf_out="${OUTPUT_DIR}/${paper_subdir}__${stem}"
  mkdir -p "${pdf_out}"

  echo ""
  echo "=================================================================="
  echo "[${COUNT}/${TOTAL}] ${pdf_out##*/}"
  echo "PDF: ${pdf}"
  echo "输出: ${pdf_out}"
  echo "=================================================================="

  if python "${PDF2ZH_ROOT}/pdf_to_zh_md.py" "${pdf}" \
      -o "${pdf_out}" \
      --model "${OPENROUTER_MODEL}" \
      --base-url "${OPENROUTER_BASE_URL}" \
      --api-key "${OPENROUTER_API_KEY}" \
      --workers 8 ; then
    SUCCESS_LIST+=("${stem}")
  else
    FAIL_LIST+=("${stem}")
    echo "[WARN] 翻译失败：${stem}（已记入 FAIL_LIST）"
  fi
done

# ---------- 汇总 ----------
echo ""
echo "=================================================================="
echo "翻译完成"
echo "=================================================================="
echo "总计：${TOTAL} 篇"
echo "成功：${#SUCCESS_LIST[@]} 篇"
echo "失败：${#FAIL_LIST[@]} 篇"
if [ "${#FAIL_LIST[@]}" -gt 0 ]; then
  echo ""
  echo "失败清单："
  for f in "${FAIL_LIST[@]}"; do
    echo "  - ${f}"
  done
fi