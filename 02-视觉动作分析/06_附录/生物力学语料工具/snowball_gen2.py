#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
snowball_gen2.py — 第二代引文追溯

为什么要做第二代：
    第八轮（V9e）从「4 篇已知撞车工作」滚出一轮，捞到 `1710.06513`（pose grammar, 2017），
    **证明式子检索会漏**。但那一轮的种子是**第一代**撞车工作，
    滚出来的 `1710.06513` 自己**只追了一层**就没管了。
    既然滚雪球刚证明有效，就该把它做到底 —— **从新撞车工作再滚一代**。

第二代种子（含第一代，但重点是新增的三个）：
    第一代：2402.17192 / TMRB 2026 / 2607.02796 / 2608.03127 / 1710.06513
    **新增**：
      · `1710.06513` 的 AAAI 正式版 DOI（有 DOI 才能走前向）
      · `2609.18667`（117 项综述，语料三内已有 PDF）
      · `2601.09949`（Kinematic Tokenization，占用方法名的那篇）

技术约束（都是实测出来的，别假设）：
    · **前向只有 OpenCitations COCI 可用**（Semantic Scholar 本机 IP 硬 429）
    · OpenCitations **只吃有 DOI 的条目** → 纯 arXiv 预印本**走不了前向**
    · 纯 arXiv 只能走后向：下 PDF → 抽参考文献区 → 用修正后的过滤器筛
    · 反爬一律不绕过
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
UA = 'kb-research/1.0 (mailto:kb-research@example.org)'

# 有 DOI 的 → 走前向
DOI_SEEDS = {
    '10.1609/aaai.v32i1.12270': 'Learning Pose Grammar（AAAI 2018 正式版）',
    '10.1109/TMRB.2025.3605962': 'TMRB 2026 可微生物力学',
    '10.1007/978-3-031-72943-0_27': 'VQ-HPS 向量量化人体姿态',
    '10.1109/cvpr46437.2021.00847': 'PoseAug 可微姿态增广',
    '10.3390/bioengineering10050510': 'Dasgupta 2023 OMC/IMC 建模',
}
# 纯 arXiv 无 DOI → 只能走后向（下 PDF 抽参考文献）
ARXIV_SEEDS = {
    '1710.06513': 'Learning Pose Grammar（arXiv 版）',
    '2608.03127': 'DigitCode 符号化手部运动',
    '2609.18667': '117 项临床/康复 MMC 综述',
    '2601.09949': 'Kinematic Tokenization（金融，占用方法名）',
}

INTERSECTION = re.compile(
    r'vocabular|codebook|tokeniz|tokenis|discrete symbolic|symbolic represent|'
    r'kinematic token|motion token|language model|neural machine translation|'
    r'vector quanti|lexical|grammar|\bVQ\b|exemplar', re.I)
UPPER = re.compile(r'wrist|elbow|shoulder|forearm|upper[- ]limb|upper extremit|hand', re.I)
VISION = re.compile(r'monocular|single camera|rgb-?d|markerless|video[- ]based|vision|visual', re.I)
# 第八轮修过的：题名里没有 vision 字样但本身就是视觉任务（"3D pose estimation"）
VISUAL_TASK = re.compile(r'pose estimat|pose recover|pose detect|keypoint|3d human|markerless', re.I)


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
    return {'doi': doi, 'title': (m.get('title') or ['?'])[0],
            'year': ((m.get('issued') or {}).get('date-parts') or [[None]])[0][0]}


def flags(title):
    f = ''
    if INTERSECTION.search(title):
        f += ' [交集机制]'
    if UPPER.search(title):
        f += ' [上肢/腕肘肩/手]'
    if (VISION.search(title)) or (VISUAL_TASK.search(title)):
        f += ' [视觉]'
    return f


def forward(doi, sleep):
    t = get('https://opencitations.net/index/coci/api/v1/citations/' + urllib.parse.quote(doi))
    if not t:
        return []
    try:
        return [x.get('citing') for x in json.loads(t) if x.get('citing')]
    except Exception:                                            # noqa: BLE001
        return []


REF_RE = re.compile(r'\[?\d{1,3}\]?\s*([A-Z][^[]{20,300}?)(?=\[\d+\]|\(\d{4}\)|$)')


def backward_from_pdf(pdf_path, label):
    raw = subprocess.run(['pdftotext', '-q', str(pdf_path), '-'],
                         capture_output=True, timeout=150).stdout.decode('utf-8', 'ignore')
    t = ' '.join(raw.split())
    up = t.upper()
    i = max(up.rfind('REFERENCES'), up.rfind('BIBLIOGRAPHY'), up.rfind('REFERENCE LIST'))
    if i < 0:
        return [], 0
    ref = t[i:i + 90000]
    out = []
    for m in REF_RE.finditer(ref):
        s = ' '.join(m.group(1).split())
        if INTERSECTION.search(s) and (UPPER.search(s) or VISION.search(s) or VISUAL_TASK.search(s)):
            out.append(s[:280])
    return out, len(REF_RE.findall(ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'snowball_gen2.json'))
    ap.add_argument('--sleep', type=float, default=1.1)
    ap.add_argument('--skip-forward', action='store_true')
    args = ap.parse_args()

    rep = {'purpose': '第二代引文追溯：从第一代新捞出的撞车工作再滚一代',
           'note': '前向仅 OpenCitations COCI（只吃有 DOI 的）；纯 arXiv 只能后向下 PDF 抽参考文献'}

    if not args.skip_forward:
        print('=' * 100)
        print('【前向】谁引用了有 DOI 的第一/二代种子')
        rep['forward'] = []
        seen = set()
        for doi, why in DOI_SEEDS.items():
            ds = forward(doi, args.sleep)
            print('─' * 96)
            print('种子 %s（%s）被引 %d' % (doi, why, len(ds)))
            recs = []
            for d in ds:
                if d in seen:
                    continue
                seen.add(d)
                m = cr_meta(d)                                 # noqa: PERF401
                time.sleep(args.sleep)
                fl = flags(m.get('title', ''))
                if fl:
                    print('   ⚑ %-60s %s%s' % (m.get('title', '?')[:60], m.get('year', ''), fl))
                recs.append({**m, 'seed': doi, 'seed_why': why, 'flags': fl.strip()})
            hit = [r for r in recs if r['flags']]
            print('   标记命中 %d / 共 %d' % (len(hit), len(recs)))
            rep['forward'].append({'seed': doi, 'why': why, 'n_cited': len(ds),
                                   'resolved': len(recs), 'flagged': hit, 'all': recs})

    print()
    print('=' * 100)
    print('【后向】纯 arXiv 种子：下 PDF 抽参考文献区')
    rep['backward'] = []
    tmp = HERE / '_tmp_gen2'
    tmp.mkdir(exist_ok=True)
    for aid, why in ARXIV_SEEDS.items():
        p = tmp / ('%s.pdf' % aid)
        try:
            p.write_bytes(urllib.request.urlopen(
                urllib.request.Request('https://arxiv.org/pdf/' + aid, headers={'User-Agent': UA}),
                timeout=150).read())
        except Exception as e:                                    # noqa: BLE001
            print('  下 PDF 失败 %s：%s' % (aid, e))
            continue
        refs, total = backward_from_pdf(p, why)
        print('─' * 96)
        print('  %s（%s）参考文献 %d 条，命中交集机制 %d' % (aid, why, total, len(refs)))
        for r in refs:
            print('     ⚑ %s' % r[:190])
        rep['backward'].append({'arxiv_id': aid, 'why': why, 'n_refs': total,
                                'flagged': refs})
        p.unlink(missing_ok=True)
    try:
        tmp.rmdir()
    except OSError:
        pass

    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    fwd_hit = sum(len(x['flagged']) for x in rep.get('forward', []))
    bwd_hit = sum(len(x['flagged']) for x in rep.get('backward', []))
    print()
    print('前向标记命中 %d；后向命中 %d' % (fwd_hit, bwd_hit))
    print('已写出：%s' % args.out)


if __name__ == '__main__':
    main()
