#!/usr/bin/env bash
# 为语料二（Transformer 发展线）跑 MinerU 英文抽取。
#
# 与 run_extract_missing.sh 的隔离点：
#   - 清单用 tf_manifest.py（从 index.json 生成），不是 paper_manifest.py
#   - 产物落 01_参考论文/运动理解/Transformer_发展线/_translation_outputs/，**不是**主语料的 _translation_outputs/
#   - todo 清单 / 日志 / 状态文件全部用独立文件名，不覆盖 /tmp/mineru_*.{log,tsv}
#   - 单进程串行，CPU-only 机器上不与他人任务抢核
set -uo pipefail

PDF2ZH_ROOT="/Users/xbpd/Projects/pdf2zh"
TF_TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PDF2ZH_ROOT}/.venv/bin/python"
OUT_ROOT="/Users/xbpd/Projects/视觉位移判断/02-视觉动作分析/01_参考论文/运动理解/Transformer_发展线/_translation_outputs"
TODO="/tmp/tf_mineru_todo.tsv"
LOG="/tmp/tf_mineru_extract.log"

cd "${PDF2ZH_ROOT}"
# shellcheck disable=SC1091
source .venv/bin/activate

"${PY}" - > "${TODO}" <<'PYEOF'
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    "tfm", Path("/Users/xbpd/Projects/视觉位移判断/02-视觉动作分析/06_附录/Transformer语料工具/翻译/tf_manifest.py"))
tfm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tfm)
for r, slug in tfm.slugs():
    if any(p for p in tfm.NEW_EXTRACT_DIR.glob(slug + "/**/*.md")):
        continue
    print(f"{r['name']}\t{slug}\t{r['pdf']}")
PYEOF

TOTAL=$(wc -l < "${TODO}" | tr -d ' ')
echo "== 语料二 MinerU 抽取：待处理 ${TOTAL} 篇（产物 ${OUT_ROOT}）==" | tee -a "$LOG"
[ "${TOTAL}" -eq 0 ] && { echo "全部已抽取"; exit 0; }

# 2026-10-01：共享工具 pdf_to_zh_md.py 的 --ocr 选项已由 {api} 改为
#   {precision, flash, local}。本脚本按可用性自动选通道：
#     precision = 云 API v4，需 MINERU_API_KEY（质量最好）
#     flash     = 云 API 免 Token（≤10MB/≤20页）
#     local     = 本机 mineru CLI（CPU-only 机器上极慢，仅兜底）
#   注：2026-10-01 实测 flash 通道对全部 8 篇返回 413/400，疑似服务端已停用该通道；
#   若三者都不可用，本脚本会明确报错而不是静默产出空文件。
OCR_MODE="local"
if [ -n "${MINERU_API_KEY:-}" ]; then
  OCR_MODE="precision"
else
  echo "!! 未检测到 MINERU_API_KEY，回退到 --ocr flash（免 Token 通道，2026-10-01 实测该通道 413/400）" | tee -a "$LOG"
  OCR_MODE="flash"
fi

i=0
while IFS=$'\t' read -r name slug pdf; do
  i=$((i+1))
  out="${OUT_ROOT}/${slug}"
  echo "" | tee -a "$LOG"
  echo "[${i}/${TOTAL}] ${name}  [ocr=${OCR_MODE}]" | tee -a "$LOG"
  if ! "${PY}" "${PDF2ZH_ROOT}/pdf_to_zh_md.py" "${pdf}" \
        -o "${out}" --ocr "${OCR_MODE}" --skip-translate 2>&1 | tail -3 | tee -a "$LOG"; then
    :
  fi
  found=$(find "${out}" -name "*.md" -type f -size +0 2>/dev/null | head -1)
  if [ -n "${found}" ]; then
    echo "  OK: ${found} ($(wc -c < "${found}" | tr -d ' ') B)" | tee -a "$LOG"
  else
    echo "  ✗ 无 md 产物：${name}" | tee -a "$LOG"
  fi
done < "${TODO}"

echo "" | tee -a "$LOG"
echo "== 语料二 MinerU 抽取结束 ==" | tee -a "$LOG"
