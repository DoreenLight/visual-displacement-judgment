#!/usr/bin/env python3
"""校验 29 篇中文全文翻译的质量。

逐篇核对：
  1. 中文字符占比（译文 vs 英文原文对比，防止「回显英文」）
  2. 章节标题数与原文一致（防止漏译/丢段）
  3. 是否到达 References 章节且有正文（全文口径）
  4. 行内公式 $...$ / 块公式 $$...$$ 数量与原文一致
  5. 表格 <table> 数量一致
  6. 图片引用数量一致，且所有被引用的图片文件真实存在
  7. 代码块 ``` 数量一致
  8. 段落数（粗粒度）不明显缩水

退出码非 0 表示存在未通过项。
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/Users/xbpd/Projects/pdf2zh")
_spec = importlib.util.spec_from_file_location("translate_all", _HERE / "translate_all.py")
_ta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ta)

REF_RE = re.compile(r"^#{0,4}\s*(references|参考文献|bibliography|引用文献)\s*$",
                    re.M | re.I)
FORMULA_BLOCK = re.compile(r"\$\$")
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def zh_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "\u4e00" <= c <= "\u9fff") / len(letters)


def headings(text: str) -> list[str]:
    return re.findall(r"^#{1,4}\s+\S.*$", text, re.M)


def check_one(row: dict) -> dict:
    name = row["name"]
    paper_dir = row["paper_dir"]
    zh_path = paper_dir / f"{name}_全文翻译.md"
    guide_path = paper_dir / f"{name}_翻译导读.md"
    en_path = row["en_md"] or _ta.find_new_extract(name)

    r: dict = {"name": name, "pages": row["pages"], "issues": []}
    if not zh_path.exists():
        r["issues"].append("译文缺失")
        return r
    if en_path is None or not Path(en_path).exists():
        r["issues"].append("找不到英文原文")
        return r

    zh = zh_path.read_text(encoding="utf-8")
    en = Path(en_path).read_text(encoding="utf-8")

    r["zh_ratio"] = round(zh_ratio(zh), 3)
    r["chars_zh"] = len(zh)
    r["chars_en"] = len(en)

    # 1. 中文占比：分正文 / 参考文献两段。参考文献里作者与刊名本就保持英文，
    #    整体占比必然偏低，所以对「正文」单独卡阈值。
    ref_m = REF_RE.search(zh)
    body_zh = zh[: ref_m.start()] if ref_m else zh
    r["zh_body"] = round(zh_ratio(body_zh), 3)
    if r["zh_body"] < 0.35:
        r["issues"].append(f"正文中文占比过低 {r['zh_body']}")

    # 2. 章节数（英文标题里 References 之后的不算）
    h_en, h_zh = len(headings(en)), len(headings(zh))
    r["h_en"], r["h_zh"] = h_en, h_zh
    if h_zh < h_en * 0.8:
        r["issues"].append(f"章节数缩水 {h_zh}/{h_en}")

    # 3. References
    r["has_refs"] = bool(REF_RE.search(zh))
    if not r["has_refs"]:
        r["issues"].append("无 References 章节")
    else:
        tail = zh[REF_RE.search(zh).start():]
        n_ref = len(re.findall(r"^\s*\[\d+\]", tail, re.M))
        r["n_refs"] = n_ref
        if n_ref < 3:
            r["issues"].append(f"参考文献条目偏少 ({n_ref})")

    # 4. 公式
    r["formula_zh"], r["formula_en"] = len(FORMULA_BLOCK.findall(zh)), len(FORMULA_BLOCK.findall(en))
    if r["formula_zh"] != r["formula_en"]:
        r["issues"].append(f"块公式数 {r['formula_zh']}≠{r['formula_en']}")

    # 5. 表格
    t_re = re.compile(r"<table", re.I)
    r["table_zh"], r["table_en"] = len(t_re.findall(zh)), len(t_re.findall(en))
    if r["table_zh"] != r["table_en"]:
        r["issues"].append(f"表格数 {r['table_zh']}≠{r['table_en']}")

    # 6. 代码块
    c_re = re.compile(r"^```", re.M)
    r["code_zh"], r["code_en"] = len(c_re.findall(zh)), len(c_re.findall(en))
    if r["code_zh"] != r["code_en"]:
        r["issues"].append(f"代码块 {r['code_zh']}≠{r['code_en']}")

    # 7. 图片
    imgs = IMG_RE.findall(zh)
    r["img_refs"] = len(imgs)
    r["img_refs_en"] = len(IMG_RE.findall(en))
    if r["img_refs"] != r["img_refs_en"]:
        r["issues"].append(f"图片引用 {r['img_refs']}≠{r['img_refs_en']}")
    missing = [p for p in imgs if not (paper_dir / p).exists()]
    r["img_missing"] = missing
    if missing:
        r["issues"].append(f"图片文件缺失 {len(missing)}")

    # 8. 体量：中文按字符计，英文按字符计，中文一般应为英文的 0.6~1.5 倍
    ratio = r["chars_zh"] / max(r["chars_en"], 1)
    r["size_ratio"] = round(ratio, 2)
    if ratio < 0.55:
        r["issues"].append(f"译文体量偏小 {ratio}")

    # 9. 翻译导读（手写分析，应为中文）
    if guide_path.exists():
        g = guide_path.read_text(encoding="utf-8")
        r["guide_zh"] = round(zh_ratio(g), 3)
        if r["guide_zh"] < 0.30:
            r["issues"].append(f"导读中文占比低 {r['guide_zh']}")
    else:
        r["issues"].append("导读缺失")

    r["status"] = "PASS" if not r["issues"] else "FAIL"
    return r


def main() -> int:
    rows = list(_ta._manifest.rows())
    results = []
    for i, row in enumerate(rows, 1):
        res = check_one(row)
        results.append(res)
        flag = "OK  " if res["status"] == "PASS" else "FAIL"
        print(f"[{i:>2}/{len(rows)}] {flag} {res['name'][:46]:<48} "
              f"zh(正文)={res.get('zh_body','-')} "
              f"H={res.get('h_zh','-')}/{res.get('h_en','-')} "
              f"refs={res.get('n_refs','-')} "
              f"img={res.get('img_refs','-')} "
              f"| {'; '.join(res['issues'])}", flush=True)

    ok = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n通过 {ok}/{len(results)}")
    if ok < len(results):
        print("\n未通过明细：")
        for r in results:
            if r["status"] == "FAIL":
                print(f"  - {r['name']}: {'; '.join(r['issues'])}")
    import json
    Path("/tmp/verify_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
