#!/usr/bin/env bash
# 为缺英文抽取的 10 篇跑 MinerU 抽取（--skip-translate，只出英文 md + images）
# 复用 paper_manifest.py 的清单，产物落 01_参考论文/_translation_outputs/<slug>/
set -uo pipefail

PDF2ZH_ROOT="/Users/xbpd/Projects/pdf2zh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PDF2ZH_ROOT}/.venv/bin/python"
OUT_ROOT="/Users/xbpd/Projects/视觉位移判断/02-视觉动作分析/01_参考论文/_translation_outputs"
LOG="/tmp/mineru_extract.log"

cd "${PDF2ZH_ROOT}"
source .venv/bin/activate

# 稳定 slug：<arxiv-ish 或短标识>__<论文名>
# 直接用「目录名__论文名」与 run_batch_cli.sh 风格一致
"${PY}" - <<'PYEOF' > /tmp/mineru_todo.tsv
import sys, importlib.util
spec = importlib.util.spec_from_file_location("m", "/Users/xbpd/Projects/视觉位移判断/02-视觉动作分析/06_附录/pdf2zh调用脚本/paper_manifest.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for r in m.rows():
    if r["en_md"] is None or not r["en_md"].exists():
        print(f'{r["name"]}\t{r["pdf"]}')
PYEOF

TOTAL=$(wc -l < /tmp/mineru_todo.tsv | tr -d ' ')
echo "== MinerU 抽取：${TOTAL} 篇 ==" | tee -a "$LOG"

i=0
while IFS=$'\t' read -r name pdf; do
  i=$((i+1))
  slug="$(echo "$name" | tr '/ ' '__')"
  out="${OUT_ROOT}/${slug}"
  echo "" | tee -a "$LOG"
  echo "==================================================================" | tee -a "$LOG"
  echo "[${i}/${TOTAL}] ${name}" | tee -a "$LOG"
  echo "  out: ${out}" | tee -a "$LOG"
  echo "==================================================================" | tee -a "$LOG"
  if "${PY}" "${PDF2ZH_ROOT}/pdf_to_zh_md.py" "${pdf}" \
      -o "${out}" --ocr api --skip-translate 2>&1 | tee -a "$LOG" | tail -3; then
    :
  fi
  # 产物实际路径
  found=$(find "${out}" -name "*.md" -type f 2>/dev/null | head -1)
  if [ -n "$found" ]; then
    sz=$(wc -c < "$found" | tr -d ' ')
    zh=$(grep -o '[一-龥]' "$found" 2>/dev/null | wc -l | tr -d ' ')
    echo "  OK: ${found} (${sz} B, 中文 ${zh})" | tee -a "$LOG"
  else
    echo "  ✗ 失败：无 md 产物" | tee -a "$LOG"
  fi
done < /tmp/mineru_todo.tsv

echo "" | tee -a "$LOG"
echo "== MinerU 抽取结束 ==" | tee -a "$LOG"
