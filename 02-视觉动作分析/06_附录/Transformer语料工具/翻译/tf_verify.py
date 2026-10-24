#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语料二（Transformer 发展线）翻译质量校验 —— 复用 verify_translations.check_one，不修改它。

隔离点与 tf_translate.py 一致：
  - 复用 `check_one()` 做逐篇质量核对（中文占比 / 编号章节 / References / 公式 / 表格 / 代码块 /
    图片引用落盘 / 拒答套话 / 行内公式缩水）
  - **不调用 main()**：它会写共享的 /tmp/verify_report.json
  - 本驱动写 /tmp/tf_verify_report.json
  - 清单走 tf_manifest，只校验语料二的 85 篇

用法：python3 tf_verify.py [--only 子串] [--thread T1]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent.parent / "pdf2zh调用脚本"
PDF2ZH_ROOT = Path("/Users/xbpd/Projects/pdf2zh")
sys.path.insert(0, str(PDF2ZH_ROOT))
sys.path.insert(0, str(HERE))

MY_CACHE = Path("/tmp/zh_translate_cache_tf")   # 与 tf_translate.py 同一隔离缓存
MY_REPORT = Path("/tmp/tf_verify_report.json")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--thread", default=None)
    args = ap.parse_args()

    tfm = _load(HERE / "tf_manifest.py", "tf_manifest")
    ver = _load(SHARED / "verify_translations.py", "verify_shared")

    # 关键：verify_translations.py 在其内部又 load 了一份 translate_all（变量名 `_ta`），
    # 那份的 `_manifest` 指向**主语料**，于是 find_new_extract 会去主语料抽取目录里找，
    # 结果整批报「找不到英文原文」。必须打的是 ver._ta，而不是本驱动自己 load 的实例。
    ver._ta._manifest = tfm
    ver._ta.CACHE_DIR = MY_CACHE

    rows = list(tfm.rows())
    if args.thread:
        rows = [r for r in rows if r.get("thread") == args.thread]
    if args.only:
        rows = [r for r in rows if args.only in r["name"]]

    results = []
    for i, row in enumerate(rows, 1):
        res = ver.check_one(row)
        res["thread"] = row.get("thread")
        res["arxiv_id"] = row.get("arxiv_id")
        results.append(res)
        flag = "OK  " if res["status"] == "PASS" else "FAIL"
        print(f"[{i:>2}/{len(rows)}] {flag} {res['name'][:44]:<46} "
              f"zh(正文)={res.get('zh_body','-')} "
              f"H={res.get('h_zh','-')}/{res.get('h_en','-')} "
              f"refs={res.get('n_refs','-')} "
              f"img={res.get('img_refs','-')} "
              f"| {'; '.join(res['issues'])}", flush=True)
        MY_REPORT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n通过 {ok}/{len(results)}")
    if ok < len(results):
        print("\n未通过明细：")
        for r in results:
            if r["status"] == "FAIL":
                print(f"  - {r['name']}: {'; '.join(r['issues'])}")
    print(f"报告: {MY_REPORT}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
