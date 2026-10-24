#!/usr/bin/env bash
# 语料二（Transformer 发展线）全流程编排：抽取 → 翻译 → 校验，按需循环直到 85 篇齐。
#
# 为什么需要编排层：
#   抽取（云端 MinerU）与翻译（MiniMax-M3）速度差一个量级，翻译必须等抽取出料；
#   而翻译是「拿到多少翻多少」，不要求全部抽完。硬串行会浪费几小时。
#   故采用「抽取跑完前，翻译反复跑已就绪的子集」的方式收敛。
#
# 断点续跑：翻译块级 checkpoint 在 /tmp/zh_translate_cache_tf/，中断后重跑只补未译块。
# 隔离：不写共享的 /tmp/translate_report.json 与 /tmp/verify_report.json。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="/Users/xbpd/Projects/pdf2zh/.venv/bin/python"
TFROOT="/Users/xbpd/Projects/视觉位移判断/02-视觉动作分析/01_参考论文/Transformer_发展线"
WORKERS="${TF_WORKERS:-6}"
ORCH_LOG=/tmp/tf_orchestrate.log

cd "$HERE"

count_done() { find "$TFROOT" -name '*_全文翻译.md' 2>/dev/null | wc -l | tr -d ' '; }
count_extract() { find "$TFROOT/_translation_outputs" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' '; }

echo "[$(date +%H:%M:%S)] 编排启动：目标 85 篇，workers=$WORKERS" | tee -a "$ORCH_LOG"

# ---- 1) 抽取（已抽取的自动跳过，可反复跑）----
# 若已有抽取进程在跑（手工启动的），先等它结束，避免两个 MinerU 进程抢同一批论文
# 注意：这里必须用 pgrep，不能用 `ps aux | grep -q`。
# 本脚本开了 `set -o pipefail`，而 `grep -q` 命中后会立刻关闭管道，
# ps 收到 SIGPIPE 以非零码退出 → 整条管道返回非零 → if 误判为「没有抽取进程」，
# 于是并发起了第二个抽取器（2026-10-01 10:44 实测踩到）。
if pgrep -f "tf_extract.sh" > /dev/null; then
  echo "[$(date +%H:%M:%S)] 检测到已有抽取进程（pgrep $(pgrep -f tf_extract.sh | tr '\n' ' ')），等待其结束…" | tee -a "$ORCH_LOG"
  while pgrep -f "tf_extract.sh" > /dev/null; do sleep 20; done
fi
bash tf_extract.sh >> /tmp/tf_extract_outer.log 2>&1
echo "[$(date +%H:%M:%S)] 抽取结束：$(count_extract) 个目录" | tee -a "$ORCH_LOG"

# ---- 2) 翻译循环：反复跑到 85 篇齐或连续两轮零进展 ----
stale=0
for round in 1 2 3 4 5 6 7 8; do
  before=$(count_done)
  echo "[$(date +%H:%M:%S)] 翻译第 $round 轮开始（已就位 $before/85）" | tee -a "$ORCH_LOG"
  "$PY" tf_translate.py --workers "$WORKERS" 2>&1 | grep -E "OK  |SKIP_NO_EN_MD|FAIL|清理空块公式" | tee -a "$ORCH_LOG"
  after=$(count_done)
  echo "[$(date +%H:%M:%S)] 第 $round 轮结束：$after/85" | tee -a "$ORCH_LOG"
  [ "$after" -ge 85 ] && break
  if [ "$after" -eq "$before" ]; then
    stale=$((stale+1))
    [ "$stale" -ge 2 ] && { echo "[$(date +%H:%M:%S)] 连续两轮零进展，停止" | tee -a "$ORCH_LOG"; break; }
  else
    stale=0
  fi
  # 若还有未抽取的论文（抽取可能因网络失败留缺口），补抽一次
  if [ "$(count_extract)" -lt 85 ]; then
    bash tf_extract.sh >> /tmp/tf_extract_outer.log 2>&1
  fi
done

# ---- 3) 质量校验 ----
echo "[$(date +%H:%M:%S)] 质量校验开始" | tee -a "$ORCH_LOG"
"$PY" tf_verify.py 2>&1 | tee -a "$ORCH_LOG" | tail -30

# ---- 4) 同步 STATUS.md 计数块的 tf_zh（随译文产出增长，不同步会让 Case 6.5 报 FAIL）----
"$PY" - <<'PYEOF'
import re
from pathlib import Path
tfroot = Path("/Users/xbpd/Projects/视觉位移判断/02-视觉动作分析/01_参考论文/Transformer_发展线")
n = len(list(tfroot.rglob("*_全文翻译.md")))
st = tfroot.parent.parent / "STATUS.md"
t = st.read_text(encoding="utf-8")
t2 = re.sub(r"^tf_zh=\d+$", "tf_zh=%d" % n, t, flags=re.M)
if t2 != t:
    st.write_text(t2, encoding="utf-8")
    print("计数块 tf_zh 已同步为 %d" % n)
else:
    print("计数块 tf_zh 已是 %d" % n)
PYEOF

echo "[$(date +%H:%M:%S)] 编排结束：译文 $(count_done)/85" | tee -a "$ORCH_LOG"
bash "/Users/xbpd/Projects/视觉位移判断/02-视觉动作分析/06_附录/自检脚本/verify_kb.sh" 2>&1 | tail -3 | tee -a "$ORCH_LOG"
