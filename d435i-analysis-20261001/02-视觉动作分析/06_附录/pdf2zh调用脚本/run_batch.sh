#!/usr/bin/env bash
# 批量翻译 01_参考论文/ 下所有 PDF 为中文 Markdown（Web UI 模式）
#
# 流程：
#   1) 启动 pdf2zh Web UI（端口 7862）
#   2) 等待 UI 就绪
#   3) 调用 tools/test_batch_via_ui.py 把 01_参考论文/ 下所有 PDF 提交到 UI
#   4) 输出落到 01_参考论文/_translation_outputs/<论文名>/auto/
#
# 用法：
#   bash run_batch.sh
#
# 备注：
#   - .env 已在 pdf2zh/ 下配好 OPENROUTER_BASE_URL / MODEL / API_KEY（stealth/ox-alpha，免费）
#   - 后台跑 Web UI 时日志写到 /tmp/pdf2zh_webui.log
#   - batch 脚本会跑完所有 PDF 才退出，可挂在终端上等

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PDF2ZH_ROOT="/Users/xbpd/Projects/pdf2zh"
PAPERS_DIR="${PROJECT_ROOT}/01_参考论文"
OUTPUT_DIR="${PAPERS_DIR}/_translation_outputs"
LOG_FILE="/tmp/pdf2zh_webui.log"
PORT="${PDF2ZH_PORT:-7862}"
URL="http://127.0.0.1:${PORT}"

echo "== pdf2zh 批量翻译 =="
echo "项目根：${PROJECT_ROOT}"
echo "PDF2ZH：${PDF2ZH_ROOT}"
echo "论文目录：${PAPERS_DIR}"
echo "输出目录：${OUTPUT_DIR}"
echo "Web UI：${URL}"
echo ""

# ---------- 1) 启动 Web UI ----------
cd "${PDF2ZH_ROOT}"
if [ ! -d ".venv" ]; then
  echo "[ERROR] 未找到 .venv，请先双击 pdf2zh.bat 完成首次安装" >&2
  exit 1
fi

# 检测是否已经在跑
if curl -fsS --max-time 3 "${URL}" >/dev/null 2>&1; then
  echo "[*] Web UI 已在运行：${URL}"
else
  echo "[*] 启动 Web UI（后台，日志：${LOG_FILE}）..."
  source .venv/bin/activate
  export PDF2ZH_PORT="${PORT}"
  nohup python app.py >"${LOG_FILE}" 2>&1 &
  WEBUI_PID=$!
  echo "[*] PID=${WEBUI_PID}"

  # 等待就绪（最多 60s）
  echo "[*] 等待 UI 就绪..."
  for i in $(seq 1 60); do
    if curl -fsS --max-time 2 "${URL}" >/dev/null 2>&1; then
      echo "[*] ✓ UI 已就绪 (${i}s)"
      break
    fi
    sleep 1
  done

  if ! curl -fsS --max-time 2 "${URL}" >/dev/null 2>&1; then
    echo "[ERROR] UI 未在 60s 内就绪，请查看日志：${LOG_FILE}" >&2
    exit 1
  fi
fi

echo ""

# ---------- 2) 批量提交 ----------
cd "${PDF2ZH_ROOT}"

# 桥接 .env → batch 脚本需要的变量
# test_batch_via_ui.py 读 DEEPSEEK_API_KEY，但底层 OpenAI 客户端只是透传，所以 OPENROUTER key 完全可以走这条路
set +u
export DEEPSEEK_API_KEY="${OPENROUTER_API_KEY:-}"
export DEEPSEEK_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
export DEEPSEEK_MODEL="${OPENROUTER_MODEL:-stealth/ox-alpha}"
set -u

if [ -z "${DEEPSEEK_API_KEY}" ]; then
  echo "[ERROR] .env 里没有 OPENROUTER_API_KEY，请检查 ${PDF2ZH_ROOT}/.env" >&2
  exit 1
fi

# 从 .env 加载 OPENROUTER_* 三个变量
if [ -f "${PDF2ZH_ROOT}/.env" ]; then
  set -a
  # 只取 OPENROUTER_ 开头的行
  grep -E '^OPENROUTER_(BASE_URL|MODEL|API_KEY)=' "${PDF2ZH_ROOT}/.env" > ./.env.openrouter
  source ./.env.openrouter
  rm -f ./.env.openrouter
  set +a
  echo "[*] 已加载 OPENROUTER 配置：${OPENROUTER_BASE_URL} / ${OPENROUTER_MODEL}"
fi

# 替换 .env 的 DEEPSEEK_* 行（因为 app.py 和 batch 脚本读这两个）
python3 - <<'PYEOF'
import os, re
env_path = "/Users/xbpd/Projects/pdf2zh/.env"
with open(env_path) as f:
    lines = f.readlines()
new_lines = []
for line in lines:
    if re.match(r'^DEEPSEEK_(API_KEY|BASE_URL|MODEL)=', line):
        continue
    new_lines.append(line)
new_lines.append(f"DEEPSEEK_API_KEY={os.environ['OPENROUTER_API_KEY']}\n")
new_lines.append(f"DEEPSEEK_BASE_URL={os.environ['OPENROUTER_BASE_URL']}\n")
new_lines.append(f"DEEPSEEK_MODEL={os.environ['OPENROUTER_MODEL']}\n")
with open(env_path, 'w') as f:
    f.writelines(new_lines)
print("[*] .env 已写入 DEEPSEEK_* = OPENROUTER_* 桥接")
PYEOF

echo ""
echo "[*] 开始批量提交 ${PAPERS_DIR} 下的 PDF..."
mkdir -p "${OUTPUT_DIR}"

# 提交时把输出目录切到 OUTPUT_DIR，这样翻译出来的 _zh.md 落在 _translation_outputs/
python3 tools/test_batch_via_ui.py "${PAPERS_DIR}" \
  --url "${URL}" \
  --workers 8 \
  --model "${OPENROUTER_MODEL}" \
  --base-url "${OPENROUTER_BASE_URL}"

echo ""
echo "[*] 完成。检查 ${OUTPUT_DIR}/<论文名>/auto/<论文名>_zh.md"