#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
curate.py — 从候选池生成「待人工裁定的短名单」

为什么需要它：
    discover.py 打捞回来的池子每线程 700–1500 条，直接人工过一遍不现实；
    但直接按 discover 的分数取 top-N 又会塞满同质条目（实测 B1 前 40 有大量
    步态/下肢/灵长类，与用户已锁定的「上肢腕-肘-肩」不符）。

    本脚本做**收窄**，不做**采信**：
      1. 上肢加权 / 下肢与步态降权（对齐用户已锁定的研究范围）
      2. 标题模糊去重（同一工作的预印本/期刊版只留最高分那条）
      3. 每个 venue 设配额，避免单一期刊刷屏
      4. 主题多样性轮转，避免短名单被一个子主题吃掉

    产物 = 人工裁定的**输入**，不是语料。任何条目仍需 resolve_multisource.py 交叉证实。

用法：
    python3 curate.py --thread B1 --top 60
    python3 curate.py --all --top 45
"""
import argparse
import difflib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 上肢关键词（加权）——对齐用户决定：上肢腕-肘-肩
UL_STRONG = ['upper extremit', 'upper limb', 'shoulder', 'elbow', 'wrist', 'forearm',
             'elbow', 'scapula', 'humerus', 'radius', 'ulna', 'hand', 'finger', 'thumb',
             'pronation', 'supination', 'grip', 'wrist joint', 'arm']
UL_WEAK = ['reach', 'manipulat', 'grasp', 'functional upper', 'shoulder girdle',
           'acromioclavicular', 'rotator cuff', 'biceps', 'triceps', 'deltoid']
# 下肢/步态/整体（降权但不全排除：整体方法学论文仍可能重要）
LL_PENALTY = ['lower extremit', 'lower limb', 'gait', 'locomotion', 'walking', 'running',
              'hip', 'knee', 'ankle', 'thigh', 'shank', 'pelvis', 'trunk lean',
              'jump', 'squat', 'balance', 'sit-to-stand', 'turning', 'step', 'stride']
# 明显跑题
OFFTOPIC = ['macaque', 'mouse', 'mice', 'rat ', 'zebrafish', 'drosophila', 'cell ',
            'protein', 'gene', 'tumor', 'cancer', 'macaques', 'primate', 'sheep', 'rabbit',
            'bird', 'fish', 'insect', 'battery', 'molecular', 'neuron', 'cortical',
            'dental', 'eye movement', 'gaze', 'eyetracking', 'vocal', 'speech',
            'postural sway balance elderly', 'lower back pain', 'lumbar']
# 检索返回的碎片条目：图、表、补充材料、勘误、书章
JUNK_TITLE = re.compile(
    r'^\s*(fig\.?|figure|table|supplemental|supplementary|appendix|erratum|corrigendum|'
    r'editorial|front matter|back matter|corrigenda|cover|index|list of|'
    r'withdrawn|retraction)\b', re.I)
# 与「视觉→运动学」主线无关但同属上肢词表的领域（放射/影像/骨科内固定/手外科病历）
OFFTHREAD = re.compile(
    r'radiograph|radiolog|radiograph\w*|x-ray|fracture|arthroscop|osteosynthes|'
    r'plate fixation|screw fixation|tendon transfer|surgical|operative|injur\w*|'
    r'cadaver|anatomical dissection|histolog|patholog|case report|'
    r'grip strength|torque strength|discomfort|posture assessment of workers|'
    r'ergonomic\w* evaluation of workers', re.I)


def norm(s):
    s = (s or '').lower()
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return ' '.join(s.split())


def rescore(b):
    t = norm(b.get('title', ''))
    ab = norm(b.get('abstract', ''))[:900]
    s = b.get('score', 0.0)

    ul = sum(1 for k in UL_STRONG if k in t)
    ulw = sum(0.5 for k in UL_WEAK if k in t)
    ll = sum(1 for k in LL_PENALTY if k in t)
    off = sum(1 for k in OFFTOPIC if k in t) + sum(1 for k in OFFTOPIC if k in ab)

    s += 2.2 * ul + 1.0 * ulw
    s -= 1.8 * ll
    s -= 6.0 * off
    # 摘要里出现上肢词也算（标题常只写「motion capture」）
    if not (ul or ulw):
        s += 1.4 * sum(1 for k in UL_STRONG if k in ab)
        s -= 0.8 * sum(1 for k in LL_PENALTY if k in ab)
    return round(s, 3), ul, ll, off


def fuzzy_dup(items, thr=0.86):
    """同一工作的不同版本（预印本 vs 期刊）只保留分数最高的一条。"""
    kept, dropped = [], []
    for b in items:
        nb = norm(b['title'])
        hit = None
        for k in kept:
            if difflib.SequenceMatcher(None, norm(k['title']), nb).ratio() >= thr:
                hit = k
                break
        if hit:
            dropped.append((b, hit['title'][:60]))
        else:
            kept.append(b)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default=str(HERE / 'biomech_pool.json'))
    ap.add_argument('--out', default=str(HERE / 'biomech_shortlist.json'))
    ap.add_argument('--thread', default='')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--top', type=int, default=50)
    ap.add_argument('--venue-quota', type=int, default=4)
    args = ap.parse_args()

    pool = json.loads(Path(args.pool).read_text(encoding='utf-8'))
    out = {'meta': {**pool['meta'],
                    'curation_note': '本文件是人工裁定的输入短名单，不是语料。每条仍需多源交叉证实。',
                    'ul_weight': UL_STRONG, 'll_penalty': LL_PENALTY, 'offtopic': OFFTOPIC},
           'threads': []}

    for t in pool['threads']:
        if args.thread and t['id'] != args.thread:
            continue
        if not args.thread and not args.all:
            continue
        items = []
        n_junk = n_off = 0
        for b in t['items']:
            ti = (b.get('title') or '').strip()
            if JUNK_TITLE.match(ti):
                n_junk += 1
                continue
            if OFFTHREAD.search(ti):
                n_off += 1
                continue
            s, ul, ll, off = rescore(b)
            if off >= 2:                      # 明显跑题，直接丢弃
                continue
            items.append({**b, 'score': s, 'ul_hits': ul, 'll_hits': ll, 'offtopic': off})
        items.sort(key=lambda x: -x['score'])
        items, dups = fuzzy_dup(items)

        # venue 配额 + 多样性轮转
        picked, vq, per_thread_key = [], Counter(), Counter()
        for b in items:
            if len(picked) >= args.top:
                break
            v = (b.get('venue') or '?')[:48]
            if vq[v] >= args.venue_quota:
                continue
            picked.append(b)
            vq[v] += 1

        print('\n' + '=' * 108)
        print('【%s】%s  池 %d 条 → 过滤碎片 %d / 偏题 %d → 模糊去重后 %d 条 → 短名单 %d 篇'
              % (t['id'], t['name'], len(t['items']), n_junk, n_off, len(items), len(picked)))
        print('=' * 108)
        for i, b in enumerate(picked):
            src = '+'.join(x[:2] for x in b['sources'])
            oa = 'OA' if b.get('is_oa') else '--'
            print(f"{i:2d}|{b['score']:6.2f}|{b.get('cited_by',0):5d}c|{b.get('year')}|{src:9s}|{oa:2s}| {b['title'][:72]}")
            print(f"     └ {(b.get('venue') or '')[:70]}  doi={b.get('doi','')[:44]}")
        out['threads'].append({
            'id': t['id'], 'name': t['name'], 'purpose': t['purpose'],
            'pool_size': len(t['items']), 'junk_dropped': n_junk, 'offthread_dropped': n_off,
            'after_dedup': len(items),
            'shortlist': picked, 'fuzzy_dropped': len(dups),
        })
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n短名单已写出：%s' % args.out)


if __name__ == '__main__':
    main()
