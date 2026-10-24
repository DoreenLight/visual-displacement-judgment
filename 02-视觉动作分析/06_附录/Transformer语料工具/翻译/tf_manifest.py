#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语料二（Transformer 发展线）85 篇的抽取/翻译清单 —— **从 index.json 生成，不手写**。

与主语料的 paper_manifest.py 完全隔离：
  - 读 01_参考论文/运动理解/Transformer_发展线/index.json（机器索引，权威口径）
  - 抽取产物落 01_参考论文/运动理解/Transformer_发展线/_translation_outputs/<slug>/
  - 不触碰 01_参考论文/_translation_outputs/（主语料抽取缓存，他人正在使用）

字段与主语料 paper_manifest.rows() 保持同构，便于 translate_all.process_paper 直接复用。
"""
from pathlib import Path

ROOT = Path("/Users/xbpd/Projects/视觉位移判断/02-视觉动作分析")
TF = ROOT / "01_参考论文" / "运动理解/Transformer_发展线"
INDEX = TF / "index.json"
# 隔离的抽取产物目录（主语料用的是 01_参考论文/_translation_outputs/，此处不共用）
NEW_EXTRACT_DIR = TF / "_translation_outputs"


def rows():
    import json
    data = json.loads(INDEX.read_text(encoding="utf-8"))
    for p in data["papers"]:
        pdf = ROOT / p["pdf_path"]
        if not pdf.exists():
            continue
        yield {
            "name": pdf.stem,              # = 目标 _全文翻译.md 的 stem
            "pdf": pdf,
            "paper_dir": pdf.parent,
            "en_md": None,                  # 一律走 find_new_extract()，只在本语料抽取目录里找
            "pages": 0,                     # MinerU 产物落地后由校验脚本回填
            "thread": p["thread"],
            "arxiv_id": p["arxiv_id"],
            "title": p["title"],
        }


def slugs():
    """抽取 slug：**必须以论文名开头**，arXiv ID 放后面。

    原因：translate_all.find_new_extract() 用 `root.glob(f"{name}*/**/*.md")`
    做**前缀**匹配（主语料的 run_extract_missing.sh 目录名就是 <论文名>，故其 glob 能命中）。
    若目录名以 arXiv ID 开头就匹配不上，翻译会误判为「缺英文抽取」。
    """
    for r in rows():
        yield r, (r["name"] + "__" + r["arxiv_id"].split("v")[0])


if __name__ == "__main__":
    rs = list(rows())
    print("语料二清单条目: %d" % len(rs))
    print("抽取产物目录: %s" % NEW_EXTRACT_DIR)
    miss = [r["name"] for r in rs if not r["pdf"].exists()]
    print("PDF 缺失: %s" % (miss or "无"))
    print("已有 _全文翻译.md: %d" % sum(1 for r in rs
          if (r["paper_dir"] / (r["name"] + "_全文翻译.md")).exists()))
    print("已有抽取 md: %d" % sum(1 for r in rs
          if any(p for p in NEW_EXTRACT_DIR.glob(r["name"] + "/**/*.md"))))
