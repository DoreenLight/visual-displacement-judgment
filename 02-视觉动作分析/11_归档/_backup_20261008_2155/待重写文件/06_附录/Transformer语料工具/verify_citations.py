#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_citations.py — 核对论证文档中出现的每个 arXiv ID 是否真实、且与本项目语料一致。

为什么需要这个：
  推演文档里手写了一批 arXiv ID（形如 2204.02605）。这些 ID 若来自记忆而非核验，
  就是**事实性风险**——最坏情况是把不存在的论文当作"领域已有结论"的证据。
  本脚本做两道核对：
    ① 该 ID 在 arXiv 官方 API 中是否真实存在，且标题与上下文宣称是否相符；
    ② 该 ID 是否与 index.json 中已入库条目一致（避免文档与语料库口径分叉）。

用法：python3 verify_citations.py [文档路径...]
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
IDX = (HERE / '../../01_参考论文/Transformer_发展线/index.json').resolve()
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36'

DEFAULT_DOCS = [
    HERE / '../../00_研究背景/Transformer_发展线与康复迁移推演.md',
    HERE / '../../02_创新点矩阵/Transformer前沿到上肢康复创新点矩阵.md',
]
# arXiv 旧式 ID 形如 2204.02605。必须排除 DOI（10.xxxx/yyyy）的「10.xxxx」片段：
#   后面紧跟 '/' 或数字、或前面紧跟数字/点/斜杠的，一律不视为 arXiv ID。
ID_RE = re.compile(r'(?<![\d./])([0-9]{4}\.[0-9]{4,5})(?![\d/])')


def arxiv_title(aid):
    url = 'https://export.arxiv.org/api/query?id_list=%s' % aid
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    xml = urllib.request.urlopen(req, timeout=40).read().decode('utf-8', 'replace')
    if '<entry>' not in xml:
        return None
    m = re.search(r'<entry>.*?<title>(.*?)</title>', xml, re.S)
    if not m:
        return None
    t = re.sub(r'\s+', ' ', m.group(1)).strip()
    return re.sub(r'&amp;', '&', t)


def main():
    docs = [Path(a) for a in sys.argv[1:]] or DEFAULT_DOCS
    idx = {}
    if IDX.exists():
        for p in json.loads(IDX.read_text(encoding='utf-8'))['papers']:
            idx[p['arxiv_id'].split('v')[0]] = p['title']

    seen, problems, ok = set(), [], 0
    for d in docs:
        if not d.exists():
            problems.append('文档不存在: %s' % d)
            continue
        text = d.read_text(encoding='utf-8')
        for aid in ID_RE.findall(text):
            if aid in seen:
                continue
            seen.add(aid)
            real = arxiv_title(aid)
            time.sleep(3.2)
            if real is None:
                problems.append('%s 引用了不存在的 arXiv ID: %s' % (d.name, aid))
                continue
            in_idx = idx.get(aid)
            if in_idx and in_idx.strip().lower() != real.strip().lower():
                problems.append('%s 的 %s：语料记为「%s」，arXiv 实为「%s」'
                                % (d.name, aid, in_idx, real))
            else:
                ok += 1
                print('  OK  %s  %s' % (aid, real[:72]), flush=True)

    print('\n核对 %d 个唯一 ID：%d 个通过，%d 个有问题' % (len(seen), ok, len(problems)))
    for p in problems:
        print('  ✗ %s' % p)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
