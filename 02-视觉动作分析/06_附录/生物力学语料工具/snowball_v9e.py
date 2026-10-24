#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
snowball_v9e.py — 引文追溯（V9e）

为什么必须做：
    前面 12 条 arXiv 式子检索对「上肢 ∩ 视觉直接 ∩ 离散符号化」**全部零命中**。
    但**式子检索天生捞不到两类东西**：
      ① 术语不同的邻近工作（作者不用 "vocabulary"/"token" 这些词）
      ② 通过引文关系连起来的间接工作
    所以「检索未命中」不能当「不存在」。这一步就是补这两个洞。

两个方向：
    **前向（citations）**：谁引用了 4 篇已知撞车工作 → 引用者里可能就有做交集的
    **后向（references）**：从语料三 41 份全文里抽参考文献表 → 看有没有交集工作藏在里面

数据源与实测限制（都是踩出来的，不要假设它们一直可用）：
    · OpenCitations COCI  → **可用**，给 DOI 换「谁引了它」
    · Semantic Scholar    → **本机 IP 硬 429**，本轮实测三次全失败，不可用
    · Europe PMC          → 仅覆盖生物医学类，对 CS 预印本无效
纪律：只走公开 API，不绕限流；命中候选只做**摘要级判定**，不入库。
"""
import argparse
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = (HERE / '../../01_参考论文/整合系统/生物力学线').resolve()
UA = 'kb-research/1.0 (mailto:kb-research@example.org)'
PDFTOTEXT = subprocess.run(['which', 'pdftotext'], capture_output=True).stdout.decode().strip()

# 4 篇已知撞车工作（种子）
SEEDS = {
    '10.1109/TMRB.2025.3605962': 'TMRB 2026 可微生物力学（上肢+卒中，**只到肩肘**）',
    '10.3390/bioengineering10050510': 'Dasgupta 2023 OMC/IMC 肌肉骨骼建模',
    '10.1007/978-3-031-72943-0_27': 'VQ-HPS 向量量化人体姿态（ECCV 侧先例）',
    '10.1109/cvpr46437.2021.00847': 'PoseAug 可微姿态增广',
}

# 交集判定词：命中任一即列为「需人工判」的候选
INTERSECTION = re.compile(
    r'vocabular|codebook|tokeniz|tokenis|discrete symbolic|symbolic represent|'
    r'kinematic token|motion token|language model|neural machine translation|'
    r'\bVQ\b|vector quanti|lexical|grammar', re.I)
UPPER = re.compile(r'wrist|elbow|shoulder|forearm|upper[- ]limb|upper extremit|hand', re.I)
VISION = re.compile(r'monocular|single camera|rgb-?d|markerless|video[- ]based|vision|visual', re.I)
# 2026-10-01 修正：原过滤器要求参考文献条目本身含「上肢/视觉」字样，
# 结果把 "Learning Pose Grammar to Encode Human Body Configuration for 3D Pose Estimation"
# 这类**真正相关**的条目误杀了 —— 它的题名里没有 vision/monocular 字样，
# 但 "3D pose estimation" 本身就是视觉任务。**过滤器不该按字面词筛相关性。**
VISUAL_TASK = re.compile(r'pose estimat|pose recover|pose detect|keypoint|3d human', re.I)


def get(url, timeout=30, retries=2):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode('utf-8', 'ignore')
        except Exception:                                        # noqa: BLE001
            time.sleep(1.5 * (a + 1))
    return ''


def crossref_meta(doi):
    u = 'https://api.crossref.org/works/' + urllib.parse.quote(doi, safe='')
    txt = get(u)
    if not txt:
        return {}
    try:
        m = json.loads(txt)['message']
    except Exception:                                            # noqa: BLE001
        return {}
    return {'doi': doi, 'title': (m.get('title') or ['?'])[0],
            'year': ((m.get('issued') or {}).get('date-parts') or [[None]])[0][0],
            'venue': (m.get('container-title') or ['?'])[0]}


def forward(seed_doi, sleep):
    """谁引用了 seed_doi。"""
    u = 'https://opencitations.net/index/coci/api/v1/citations/' + urllib.parse.quote(seed_doi)
    txt = get(u)
    if not txt:
        return [], 'fetch_failed'
    try:
        items = json.loads(txt)
    except Exception:                                            # noqa: BLE001
        return [], 'parse_failed'
    out = []
    for it in items:
        d = it.get('citing', '')
        if d:
            out.append(d)
    time.sleep(sleep)
    return out, 'ok'


_REF_LINE = re.compile(r'\[\d+\]|\(\d{4}\)|doi:|10\.\d{4,}/', re.I)


def backward(sleep):
    """从语料三 41 份全文抽参考文献表，筛出同时沾「上肢/视觉/交集机制」的条目。"""
    import glob
    import html as _html
    found, scanned = [], 0
    files = (glob.glob(str(CORPUS / '*' / '*.pdf')) +
             glob.glob(str(CORPUS / '*' / '*.xml')))
    for f in sorted(files):
        if f.endswith('.pdf'):
            if not PDFTOTEXT:
                continue
            try:
                raw = subprocess.run([PDFTOTEXT, '-q', f, '-'],
                                     capture_output=True, timeout=120).stdout.decode('utf-8', 'ignore')
            except Exception:                                    # noqa: BLE001
                continue
        else:
            raw = Path(f).read_text(encoding='utf-8', errors='ignore')
            raw = re.sub(r'<[^>]+>', ' ', raw)
            raw = _html.unescape(raw)
        raw = ' '.join(raw.split())
        i = raw.upper().rfind('REFERENCES')
        if i < 0:
            i = raw.upper().rfind('BIBLIOGRAPHY')
        if i < 0:
            continue
        scanned += 1
        ref = raw[i:i + 90000]                    # 参考文献区上限，避免吃进正文
        for m in re.finditer(r'\[?\d{1,3}\]?\s*([A-Z][^[]{20,300}?)(?=\[\d+\]|\(\d{4}\)|$)', ref):
            seg = m.group(1).strip()
            if not _REF_LINE.search(seg) and len(seg) < 40:
                continue
            if INTERSECTION.search(seg) and (UPPER.search(seg) or VISION.search(seg) or VISUAL_TASK.search(seg)):
                found.append({'source_file': Path(f).relative_to(CORPUS).name,
                              'ref': seg[:260]})
        time.sleep(sleep)
    return found, scanned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'snowball_v9e.json'))
    ap.add_argument('--sleep', type=float, default=1.2)
    ap.add_argument('--skip-backward', action='store_true')
    args = ap.parse_args()

    report = {'purpose': 'V9e 引文追溯：补「式子检索捞不到」的两个洞（术语不同的邻近工作 + 引文间接工作）',
              'limits': 'Semantic Scholar 本机 IP 硬 429（实测三次），前向只用 OpenCitations COCI；'
                        'OpenCitations 只覆盖有 DOI 的条目，arXiv 预印本无 DOI 走不到前向',
              'forward': [], 'backward': []}

    print('=' * 100)
    print('【前向】谁引用了 4 篇已知撞车工作')
    seen = set()
    for seed, why in SEEDS.items():
        dois, st = forward(seed, args.sleep)
        print('─' * 96)
        print('种子 %s  [%s]  被引 %d' % (seed, st, len(dois)))
        recs = []
        for d in dois:
            if d in seen:
                continue
            seen.add(d)
            m = crossref_meta(d)                 # noqa: PERF401
            time.sleep(args.sleep)
            flag = ''
            if INTERSECTION.search(m.get('title', '')):
                flag += ' [交集机制]'
            if UPPER.search(m.get('title', '')):
                flag += ' [上肢/腕肘肩/手]'
            if VISION.search(m.get('title', '')):
                flag += ' [视觉/单目]'
            print('   · %-62s %s%s' % (m.get('title', '?')[:62], m.get('year', ''), flag))
            recs.append({**m, 'seed': seed, 'seed_why': why, 'flags': flag.strip()})
        report['forward'].append({'seed': seed, 'seed_why': why, 'status': st,
                                  'n': len(dois), 'citing': recs})

    if not args.skip_backward:
        print()
        print('=' * 100)
        print('【后向】语料三全文的参考文献表里有没有交集工作')
        found, scanned = backward(args.sleep * 0.3)
        print('  扫到参考文献区的全文：%d 份；候选条目：%d' % (scanned, len(found)))
        for x in found[:40]:
            print('   · [%s] %s' % (x['source_file'][:34], x['ref'][:150]))
        report['backward'] = found
        report['backward_scanned_files'] = scanned

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    fwd = sum(x['n'] for x in report['forward'])
    print()
    print('前向命中 %d 篇（已去重 %d）；后向候选 %d 条' % (fwd, len(seen), len(report['backward'])))
    print('已写出：%s' % args.out)


if __name__ == '__main__':
    main()
