#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
snowball_gen3.py — 第三代引文追溯（种子 = 临床侧撞红扫描新坐实的 F11）

为什么做第三代：
    临床侧扫描（sweep_clinical.py）坐实了 **F11 = ACM MM 2024
    *Translating Motion to Notation: Hand Labanotation…***（`10.1145/3664647.3680568`）。
    这条被引 **6（Crossref）/ 10（Google Scholar）次**，
    **而它的引用网络我们一条都没追过**。
    前面两代已经反复证明一件事：**式子检索会漏，引文追溯兜底**（V9e 教训）。
    一篇 ACM MM 的记谱工作被引了 6–10 次，**它的引用者里极可能有康复侧的落地** ——
    而康复侧正是我们此前完全没查过的那一面。**这一刀最该补。**

与前两代的差别：
    1. 标记器**新增临床侧词表**（rehab / clinical / patient / stroke / therapy / therapist…），
       前两代只标「交集机制 / 上肢 / 视觉」，**标不到临床**。
    2. 种子额外加上 F12 相关的一篇：Barbacci 2002「Labanotation: a universal movement
       notation **language**」—— 用来查「把运动当语言」这个提法在引用网络里的扩散。

技术约束（沿用前两代实测结论，不要重新假设）：
    · 前向只有 **OpenCitations COCI** 可用（Semantic Scholar / OpenAlex 本机 IP 硬 429）
    · OpenCitations **只吃有 DOI 的条目** → 纯 arXiv 走不了前向
    · 反爬一律不绕过
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = 'kb-research/1.0 (mailto:kb-research@example.org)'

DOI_SEEDS = {
    '10.1145/3664647.3680568': 'F11 ACM MM 2024 Hand Labanotation「Translating Motion to Notation」',
    '10.22323/2.01010201': 'F12 Barbacci 2002「Labanotation: a universal movement notation language」',
    '10.1145/3664647.3680568#cites': None,          # 占位防重，见下方去重
}
DOI_SEEDS.pop('10.1145/3664647.3680568#cites')

ARXIV_SEEDS = {}

# 标记器：CV 侧沿用 + **新增临床侧**
INTERSECTION = re.compile(
    r'vocabular|codebook|tokeniz|tokenis|discrete symbolic|symbolic represent|'
    r'kinematic token|motion token|language model|neural machine translation|'
    r'vector quanti|lexical|grammar|\bVQ\b|exemplar', re.I)
UPPER = re.compile(r'wrist|elbow|shoulder|forearm|upper[- ]limb|upper extremit|hand', re.I)
VISION = re.compile(r'monocular|single camera|rgb-?d|markerless|video[- ]based|vision|visual', re.I)
VISUAL_TASK = re.compile(r'pose estimat|pose recover|pose detect|keypoint|3d human|markerless', re.I)
# ★ 本代新增：临床/康复侧。前两代标不到这一面，所以「只查了 CV 侧」的盲区才没被发现
CLINICAL = re.compile(
    r'rehabil|clinical|clinician|patient|stroke|therap|neurorehab|'
    r'upper limb function|assessment|paresis|spastic|activities of daily|'
    r'\bFMA\b|physiother|occupational therap|quality of life', re.I)
NOTATION = re.compile(r'laban|notation|notate|choreograph|entrain|symbolic', re.I)


def get(url, timeout=40, retries=2):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode('utf-8', 'ignore')
        except Exception:                                        # noqa: BLE001
            time.sleep(1.5 * (a + 1))
    return ''


def cr_meta(doi):
    t = get('https://api.crossref.org/works/' + urllib.parse.quote(doi, safe=''))
    if not t:
        return {}
    try:
        m = json.loads(t)['message']
    except Exception:                                            # noqa: BLE001
        return {}
    au = m.get('author', [])
    return {'doi': doi, 'title': (m.get('title') or ['?'])[0],
            'year': ((m.get('issued') or {}).get('date-parts') or [[None]])[0][0],
            'venue': (m.get('container-title') or [''])[0],
            'type': m.get('type', ''),
            'authors': ['%s %s' % (a.get('family', ''), a.get('given', '')) for a in au[:6]],
            'n_auth': len(au)}


def flags(title):
    f = ''
    if INTERSECTION.search(title):
        f += ' [交集机制]'
    if UPPER.search(title):
        f += ' [上肢]'
    if VISION.search(title) or VISUAL_TASK.search(title):
        f += ' [视觉]'
    if NOTATION.search(title):
        f += ' [记谱法]'
    if CLINICAL.search(title):                                    # ★ 本代新增
        f += ' [临床/康复]'
    return f


def forward(doi, sleep):
    t = get('https://opencitations.net/index/coci/api/v1/citations/' + urllib.parse.quote(doi))
    if not t:
        return []
    try:
        return [x.get('citing') for x in json.loads(t) if x.get('citing')]
    except Exception:                                            # noqa: BLE001
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'snowball_gen3.json'))
    ap.add_argument('--sleep', type=float, default=1.0)
    args = ap.parse_args()

    rep = {
        'purpose': '第三代引文追溯：种子 = 临床侧扫描新坐实的 F11（ACM MM 2024 Hand Labanotation）',
        'new_in_gen3': '标记器新增 [临床/康复] 与 [记谱法] —— 前两代标不到临床侧',
        'constraint': '前向仅 OpenCitations COCI（只吃有 DOI 的）；不绕过反爬',
        'forward': [],
    }

    for doi, why in DOI_SEEDS.items():
        citing = forward(doi, args.sleep)
        rows, flagged = [], []
        for c in citing:
            m = cr_meta(c)
            if not m:
                continue
            fl = flags(m['title'])
            m.update({'seed': doi, 'seed_why': why, 'flags': fl})
            rows.append(m)
            if fl:
                flagged.append(m)
                print('   ↳ [%s] %s' % (m['year'], m['title'][:96]))
                print('       %s | %s' % (m.get('venue', '')[:60], fl))
            time.sleep(args.sleep * 0.5)
        print('[FWD] %s  n_cited=%d  解析 %d  标记 %d' %
              (doi, len(citing), len(rows), len(flagged)))
        print('      %s' % why)
        rep['forward'].append({'seed': doi, 'why': why, 'n_cited': len(citing),
                               'resolved': len(rows), 'n_flagged': len(flagged),
                               'flagged': flagged, 'all': rows})

    # 汇总
    allrows = [r for f in rep['forward'] for r in f['all']]
    uniq = {}
    for r in allrows:
        uniq.setdefault(r['doi'], r)
    cat = {}
    for r in uniq.values():
        for fl in re.findall(r'\[([^\]]+)\]', r.get('flags') or ''):
            cat[fl] = cat.get(fl, 0) + 1
    rep['summary'] = {
        'n_seeds': len(DOI_SEEDS),
        'n_rows': len(allrows),
        'n_unique_doi': len(uniq),
        'n_flagged_unique': sum(1 for r in uniq.values() if r.get('flags')),
        'flag_histogram': cat,
    }
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n汇总：', json.dumps(rep['summary'], ensure_ascii=False, indent=1))
    print('→', args.out)


if __name__ == '__main__':
    main()
