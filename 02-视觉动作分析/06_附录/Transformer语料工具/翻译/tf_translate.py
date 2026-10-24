#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语料二（Transformer 发展线）全文翻译驱动 —— **复用 translate_all.py 的全部逻辑，但不修改它**。

为什么要复用而不是复制 28 KB：
  `translate_all.py` 是共享资产（他人正在用它跑主语料 29 篇的 --fixnums 修复）。
  复制一份必然与上游漂移；改它则会打断别人的运行。
  所以这里用 importlib 载入其模块，**只替换三个模块级变量**：
      ta.CACHE_DIR          → 隔离的翻译缓存
      ta._manifest          → 语料二的清单（rows() / NEW_EXTRACT_DIR）
  `find_new_extract()` 读的是 `_manifest.NEW_EXTRACT_DIR`，因此抽取检索也自动被重定向到语料二目录。
  **不调用 ta.main()**：它会写共享的 /tmp/translate_report.json，会覆盖别人的报告；本驱动写自己的报告。

绝不触碰（他人正在使用）：
  06_附录/pdf2zh调用脚本/translate_all.py · paper_manifest.py · verify_translations.py
  01_参考论文/_translation_outputs/ · 各主语料子目录 · /tmp/translate_report.json · /tmp/zh_translate_cache

用法：
    python3 tf_translate.py --dry-run             # 只报计划，不调 API
    python3 tf_translate.py --workers 6            # 翻译（断点续跑，安全可中断）
    python3 tf_translate.py --only T5 --dry-run    # 只看 T5 线程
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent.parent / "pdf2zh调用脚本"   # 只读引用：06_附录/pdf2zh调用脚本
PDF2ZH_ROOT = Path("/Users/xbpd/Projects/pdf2zh")

sys.path.insert(0, str(PDF2ZH_ROOT))
sys.path.insert(0, str(HERE))

# 隔离的翻译块缓存（主语料用的是 /tmp/zh_translate_cache）
MY_CACHE = Path("/tmp/zh_translate_cache_tf")
MY_REPORT = Path("/tmp/tf_translate_report.json")
MY_LOG = Path("/tmp/tf_translate.log")

# 模型偶尔会在译文里吐出一个**空**的块公式（实测：Attention Is All You Need 出现一处 `$$ $$`）。
# 空公式不含任何内容，删掉它不损失信息，但能让译文与原文的块公式数对齐（否则 tf_verify 会报
# 「块公式数 12≠10」）。只删**完全为空**的 $$ $$ / $$ \n $$，绝不碰有内容的公式。
EMPTY_MATH = re.compile(r"\$\$[ \t]*(?:\n[ \t]*)?\$\$[ \t]*(?:\n[ \t]*)?", re.M)


def strip_empty_math(path: Path) -> int:
    try:
        t = path.read_text(encoding="utf-8")
    except OSError:
        return 0
    new, n = EMPTY_MATH.subn("", t)
    if n:
        path.write_text(new, encoding="utf-8")
    return n


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只跑 name 含该子串的论文")
    ap.add_argument("--thread", default=None, help="只跑该线程（用 tf_manifest --only 配合）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk-size", type=int, default=3000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--repair", action="store_true", help="只重跑被污染的块")
    ap.add_argument("--deecho", action="store_true", help="只删英文复读段（纯本地）")
    ap.add_argument("--fixnums", action="store_true", help="重跑数字对不上的块")
    args = ap.parse_args()

    tfm = _load(HERE / "tf_manifest.py", "tf_manifest")
    ta = _load(SHARED / "translate_all.py", "translate_all_shared")

    # ---- 三处隔离替换（不触碰共享文件）----
    ta.CACHE_DIR = MY_CACHE
    ta._manifest = tfm

    rows = list(tfm.rows())
    if args.thread:
        rows = [r for r in rows if r.get("thread") == args.thread]
    if args.only:
        rows = [r for r in rows if args.only in r["name"]]
    if args.limit:
        rows = rows[: args.limit]

    if args.dry_run:
        tot = 0
        no_en = 0
        for r in rows:
            res = ta.process_paper(r, None, 0, args.chunk_size, True)
            if res.get("status") == "PLAN":
                tot += res["chunks"]
                print(f"{res['chunks']:>4} chunks ({res['todo']} 待译)  {r['name'][:56]}")
            else:
                no_en += 1
                print(f"{'—':>4} {'缺英文抽取':>12}  {r['name'][:56]}")
        print(f"\n条目 {len(rows)} ｜ 待译 chunks 合计 {tot} ｜ 缺英文抽取 {no_en}")
        return 0

    import pdf_to_zh_md
    key = pdf_to_zh_md.load_api_key()
    if not key:
        sys.exit("no API key")
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=ta.BASE_URL, timeout=180, max_retries=0)

    MY_CACHE.mkdir(parents=True, exist_ok=True)
    report = []
    if MY_REPORT.exists():                       # 断点续跑：保留历史条目
        try:
            prev = {r["name"]: r for r in json.loads(MY_REPORT.read_text(encoding="utf-8"))}
        except Exception:                        # noqa: BLE001
            prev = {}
    else:
        prev = {}

    t0 = time.time()
    for i, r in enumerate(rows, 1):
        name = r["name"]
        print(f"\n[{i}/{len(rows)}] {name}", flush=True)
        try:
            res = ta.process_paper(r, client, args.workers, args.chunk_size, False,
                                   repair=args.repair, deecho=args.deecho,
                                   fixnums=args.fixnums)
        except Exception as e:                    # noqa: BLE001
            res = {"name": name, "status": "FAIL", "error": str(e)[:300]}
        res["thread"] = r.get("thread")
        res["arxiv_id"] = r.get("arxiv_id")
        report.append(res)
        MY_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        if res["status"] == "OK":
            n_math = strip_empty_math(Path(res["out"]))
            if n_math:
                print(f"    清理空块公式 {n_math} 处", flush=True)
            print(f"    OK  zh={res['zh_ratio']}  {res['chunks']} chunks "
                  f"(本轮译 {res['translated_this_run']})  {res['seconds']}s  "
                  f"img+{res['images_copied']}", flush=True)
        else:
            print(f"    {res['status']}: {res.get('error', '')}", flush=True)

    ok = sum(1 for r in report if r["status"] == "OK")
    print(f"\n总耗时 {time.time() - t0:.0f}s ｜ 成功 {ok}/{len(report)}", flush=True)
    print(f"报告: {MY_REPORT}", flush=True)
    return 0 if ok == len(report) else 1


if __name__ == "__main__":
    sys.exit(main())
