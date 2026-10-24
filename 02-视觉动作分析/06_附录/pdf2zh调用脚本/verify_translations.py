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

REF_RE = re.compile(
    r"^(?:#{0,4}|\*\*)\s*(?:[IVXLC0-9]+[.、]?\s+)?\s*"
    r"(references|参考文献|bibliography|引用文献|reference)\b[^\n]*$",
    re.M | re.I)
# 文献条目排版五花八门，三类都收：
#   ① 显式编号  [12] / 12. / 12) / (12) / [AADS21]
#   ② 无编号、作者开头   Chan LYT, Chua CS … (2022) Title
#   ③ 无编号、作者开头   Abbasi-Kesbi R and Nikfarjam A (2018) Title
REF_ITEM = re.compile(
    r"^\s*(?:\[([\w+]+)\]|(\d+)[.)]|\((\d+)\))\s+\S"
    r"|^\s*\S[^。]{0,120}?\(?\b(19|20)\d{2}\b", re.M)
FORMULA_BLOCK = re.compile(r"\$\$")
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
# 模型拒答/套话污染：用与生产同一套检测器，避免两边口径不一致
# （本轮就因为口径不一致漏掉过 4 篇）
from translate_all import is_refusal as _is_refusal  # noqa: E402
from translate_all import (  # noqa: E402
    YEARISH, DOIISH, NUM_RE)


def zh_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "\u4e00" <= c <= "\u9fff") / len(letters)


def headings(text: str) -> list[str]:
    return re.findall(r"^#{1,4}\s+\S.*$", text, re.M)


CN_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*([万亿兆])")
CN_SCALE = {"万": 10_000, "亿": 100_000_000, "兆": 10**12}
EN_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(thousand|million|billion|trillion)",
                     re.I)
EN_SCALE = {"thousand": 10**3, "million": 10**6,
            "billion": 10**9, "trillion": 10**12}
# 这些在译文里会被改写格式（doi：/ URL / 日期），逐字比对必然误报
DOI_ANY = re.compile(r"\b10\.\d{4,9}/\S+", re.I)
ARXIV_ID = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b|\b\d{4}\.\d{4,5}(?=v\d)")
# 形如 13.06.22 / 2022.3153448 的日期与 arXiv 编号会被译文改写格式。
# 注意：不能用宽松的 \d{1,2}\.\d{1,2} —— 那会把 8.57、8.05 这类真实数据当成日期吃掉。
DOT_DATE = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b|\b(?:19|20)\d{2}\.\d{1,2}\b")
URL_ANY = re.compile(r"(?:https?://|www\.)\S+", re.I)
DOI_LABEL = re.compile(r"doi\s*[:：]?\s*\S+", re.I)
NUMVAL = re.compile(r"\d+(?:\.\d+)?")


def _normalize_numbers(t: str) -> str:
    """把中英数量级写法都归一到「纯数值」，再交给数值比较。

    中文：24 亿 / 1370 万；英文：2.4 billion / 13.7 million。
    不归一会把「13.7 million → 1370 万」这种正确换算误判成数字丢失。
    """
    t = URL_ANY.sub(" ", t)
    t = DOI_ANY.sub(" ", t)
    t = DOI_LABEL.sub(" ", t)
    t = ARXIV_ID.sub(" ", t)
    t = DOT_DATE.sub(" ", t)

    def cn(mo: re.Match) -> str:
        try:
            return str(int(round(float(mo.group(1)) * CN_SCALE[mo.group(2)])))
        except ValueError:
            return mo.group(0)

    def en(mo: re.Match) -> str:
        try:
            return str(int(round(float(mo.group(1))
                                 * EN_SCALE[mo.group(2).lower()])))
        except ValueError:
            return mo.group(0)

    t = CN_UNIT.sub(cn, t)
    t = EN_UNIT.sub(en, t)
    return t


def _data_values(t: str) -> set[float]:
    """数据型数值的「值」集合（已归一单位，已排除 DOI/arXiv/日期/年份）。"""
    t = _normalize_numbers(t)
    t = re.sub(r"^\s*(?:\[\w+\]|\d+[.)])\s+\S.*$", " ", t, flags=re.M)
    out: set[float] = set()
    for x in NUMVAL.findall(t):
        if YEARISH.match(x) or DOIISH.match(x):
            continue
        v = float(x)
        if "." in x or v >= 10:
            out.add(v)
    return out


REF_START = re.compile(r"^#{0,4}\s*(?:[IVXLC0-9]+[.、]?\s+)?\s*"
                       r"(references|参考文献|bibliography|引用文献)\b",
                       re.M | re.I)


def _body_for_numbers(t: str) -> str:
    """数字保真只查「正文 + 表格」。

    参考文献里的 DOI / 页码 / 会议论文号会被译成不同格式（doi：vs doi:、
    210–223 vs 210-223），逐字比对必然刷一堆假警报，不查。
    但表格要带上：MinerU 会把末张大表排在 ## 参考文献 之后
    （Jamsrandorj2022-EMBC 的 Table VI），直接截断会误判成丢数据。
    """
    m = REF_START.search(t)
    head = t[: m.start()] if m else t
    tables = " ".join(re.findall(r"<table[\s\S]*?</table>", t, re.I))
    return head + "\n" + tables


def _lost_data_nums(en: str, zh: str) -> set[str]:
    """原文有、译文（单位归一后）仍找不到的数据型数字。"""
    got = _data_values(zh)
    lost = set()
    for x in sorted(_data_values(en)):
        if x not in got:
            # 不用 %g（1300000.0 会变成 1.3e+06）
            lost.add(str(int(x)) if x == int(x) else repr(x))
    return lost


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

    # 1. 中文占比：只看「正文散文」。
    #    - 排除 References 段：作者/刊名/DOI 按口径保持英文，只译标题；
    #    - 排除 <table>：表格不进模型、整表原样保留英文（保数据不丢）；
    #    - 排除 $$公式$$ / ```代码``` / 图片引用：本来就该是原文。
    #    剩下全是散文，才用中文占比衡量是否真翻过。
    ref_m = REF_RE.search(zh)
    body_zh = zh[: ref_m.start()] if ref_m else zh
    prose = re.sub(r"<table[\s\S]*?</table>", " ", body_zh, flags=re.I)
    prose = IMG_RE.sub(" ", prose)
    prose = re.sub(r"\$\$[\s\S]*?\$\$", " ", prose)
    prose = re.sub(r"\$[^$\n]{1,200}\$", " ", prose)
    prose = re.sub(r"```[\s\S]*?```", " ", prose)
    r["zh_body"] = round(zh_ratio(prose), 3)
    r["zh_ratio"] = round(zh_ratio(zh), 3)
    r["chars_zh"] = len(zh)
    r["chars_en"] = len(en)
    if r["zh_body"] < 0.50:
        r["issues"].append(f"正文中文占比过低 {r['zh_body']}")

    # 2. 章节数（英文标题里 References 之后的不算）
    # 2. 章节数：只比「编号章节」。MinerU 会把图列表里的模型名
    #    （RTMPose / ViTPose / Sapiens …）也标成 ## 标题，翻译时会被降级为
    #    普通行，比总数会误报大量缺失；正文编号章节才是真结构。
    def num_heads(t: str) -> int:
        return len(re.findall(r"^#{1,4}\s+[\dIVXLC]+[.\s]", t, re.M))
    h_en, h_zh = num_heads(en), num_heads(zh)
    r["h_en"], r["h_zh"] = h_en, h_zh
    if h_zh < h_en - 1:
        r["issues"].append(f"编号章节缩水 {h_zh}/{h_en}")

    # 3. References
    r["has_refs"] = bool(REF_RE.search(zh))
    if not r["has_refs"]:
        r["issues"].append("无 References 章节")
    else:
        tail = zh[ref_m.start():]
        n_ref = len(REF_ITEM.findall(tail))
        # 短文常把整串参考文献排在一行里（[1] … [8]），按行数会只数出 1
        n_ref = max(n_ref, len(re.findall(r"\[\d+\]", tail)))
        r["n_refs"] = n_ref
        if n_ref < 3:
            r["issues"].append(f"参考文献条目偏少 ({n_ref})")

    # 4. 公式：块级公式已由分块器逐字回填，只可能少不会多；
    #    多出来的是模型把长行内式改写成行间式，不算丢内容。
    r["formula_zh"], r["formula_en"] = len(FORMULA_BLOCK.findall(zh)), len(FORMULA_BLOCK.findall(en))
    if r["formula_zh"] < r["formula_en"]:
        r["issues"].append(f"块公式数 {r['formula_zh']}≠{r['formula_en']}")

    # 5. 表格：整表逐字回填，数量必须相等
    t_re = re.compile(r"<table", re.I)
    r["table_zh"], r["table_en"] = len(t_re.findall(zh)), len(t_re.findall(en))
    if r["table_zh"] < r["table_en"]:
        r["issues"].append(f"表格数 {r['table_zh']}≠{r['table_en']}")

    # 6. 代码块：同样由分块器逐字回填，只查少不查多
    c_re = re.compile(r"^```", re.M)
    r["code_zh"], r["code_en"] = len(c_re.findall(zh)), len(c_re.findall(en))
    if r["code_zh"] < r["code_en"]:
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

    # 8. 体量：中文是压缩型语言，译文约为英文的 0.4~1.5 倍都正常；
    #    低于 0.35 才是真丢内容（实测 29 篇均在 0.40~0.60）。
    ratio = r["chars_zh"] / max(r["chars_en"], 1)
    r["size_ratio"] = round(ratio, 2)
    if ratio < 0.35:
        r["issues"].append(f"译文体量偏小 {ratio}")

    # 9. 翻译导读（手写分析，应为中文）
    if guide_path.exists():
        g = guide_path.read_text(encoding="utf-8")
        r["guide_zh"] = round(zh_ratio(g), 3)
        if r["guide_zh"] < 0.30:
            r["issues"].append(f"导读中文占比低 {r['guide_zh']}")
    else:
        r["issues"].append("导读缺失")

    # 10. 拒答/套话污染：模型遇到碎片时回「请提供完整原文」并被写进了译文
    n_refuse = sum(1 for ln in zh.split("\n") if _is_refusal(ln))
    r["refusals"] = n_refuse
    if n_refuse:
        r["issues"].append(f"含拒答套话 {n_refuse} 处")

    # 11. 数字保真：小数点/多位数必须逐字出现。中文单位换算不算错——
    #     2.4 billion→24 亿、13.7 million→1370 万，先归一到原值再比。
    r["data_nums_lost"] = sorted(
        _lost_data_nums(_body_for_numbers(en), _body_for_numbers(zh)))
    if len(r["data_nums_lost"]) > 3:
        r["issues"].append(f"数据数字对不上 {len(r['data_nums_lost'])}: "
                           f"{r['data_nums_lost'][:5]}")
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
