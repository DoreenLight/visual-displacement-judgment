#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_xai_rehab.py — 「可解释 / 可读性 × 康复运动学」深扫（第三轮临床侧扫描遗留面）

为什么要单独写这个工具：
    临床侧扫描（sweep_clinical.py）里，**E 组「可解释 AI × 康复」命中非常密集** ——
    XAI 综述、clinical decision support、clinician trust、surgical skills assessment 一大片。
    我上一轮**只判了「不撞」就放过去了**，并在日志里如实承认：
    「这一面严格说还没查过，只是扫过去了」。

    这次要回答一个**具体问题**，不是泛泛看热闹：

        Q1 康复领域**是否已经建立**「把运动离散符号化 → 让临床可解释」这条路？
           （若无 → 我们的主张 A/B/E 差异成立；若有 → 我们的 interpretability 论点被占）
        Q2 有没有人用 **grammar / 状态机 / 有限状态** 表达康复动作？
        Q3 康复领域是否已经借 **language / translation** 隐喻描述运动？

    上一轮只看了**题名**。这次对高危命中要**取摘要**再做判定 ——
    题名级判定在「可解释」这个词上污染极重（XAI 综述满地都是）。

与既有工具的分工：
    sweep_clinical.py    = 24 式广度定位（找到「哪一面有东西」）
    本工具               = 对其中最可疑的一面做**摘要级深判**

纪律（不变）：
    · Crossref 零命中不可信；只有 arXiv 零命中可作空白证据（K2）
    · 题名级 ≠ 结论级；关键条目一律取摘要
    · 不绕过反爬
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

QUERIES = [
    # —— Q1：离散符号表示 → 临床可解释 ——
    ('Q1a 离散符号化+康复',   'crossref', 'discrete symbolic representation movement clinical interpretability'),
    ('Q1b 符号化+物理治疗',   'crossref', 'symbolic representation physiotherapy movement analysis interpretable'),
    ('Q1c 可解释+运动学+康复', 'crossref', 'interpretable movement kinematics rehabilitation clinicians'),
    ('Q1d 可解释+动捕+临床',  'crossref', 'interpretable motion capture clinical assessment joint angles'),
    ('Q1e arXiv 可解释 符号 康复', 'arxiv', 'all:"rehabilitation" AND all:"symbolic" AND all:"interpretable"'),
    ('Q1f arXiv 可解释 上肢 动捕', 'arxiv', 'all:"interpretable" AND all:"upper limb" AND all:"motion capture"'),

    # —— Q2：grammar / 状态机 表达康复动作 ——
    ('Q2a grammar+康复动作',   'crossref', 'grammar representation rehabilitation exercise movement'),
    ('Q2b 状态机+康复动作',    'crossref', 'finite state machine rehabilitation movement analysis exercise'),
    ('Q2c 运动技能+语法',      'crossref', 'motor skill grammar representation human movement'),
    ('Q2d arXiv grammar 康复 动作', 'arxiv', 'all:"grammar" AND all:"rehabilitation" AND all:"movement"'),

    # —— Q3：康复领域的语言/翻译隐喻 ——
    ('Q3a 运动+语言+康复',     'crossref', 'movement as language rehabilitation motor control representation'),
    ('Q3b 翻译+运动控制',      'crossref', 'translation analogy motor control movement representation'),
    ('Q3c arXiv translation 康复 运动控制', 'arxiv', 'all:"translation" AND all:"motor control" AND all:"representation"'),

    # —— 反向：临床可读性基准/评价 ——
    ('Q4a 临床可解释性+动作',  'crossref', 'clinician interpretability rating movement analysis model output'),
    ('Q4b 康复+透明模型',      'crossref', 'transparent interpretable model physiotherapy decision support'),
    ('Q4c arXiv 临床 信任 模型 康复', 'arxiv', 'all:"clinician" AND all:"trust" AND all:"machine learning" AND all:"rehabilitation"'),
]

# 高危判定词：只有同时含「康复/临床」与「符号化机制」才算真撞
CLIN = re.compile(r'rehabil|clinical|clinician|patient|stroke|therap|physiother|neurorehab', re.I)
MECH = re.compile(
    r'discrete symbolic|symbolic represent|notation|notate|laban|'
    r'grammar|finite state|state machine|codebook|vocabular|tokeni|'
    r'\bVQ\b|vector quanti|language model|translation|lexical', re.I)
# 一般性 XAI 噪声：这些是「可解释 AI」大杂烩，不构成「符号化运动表示」的撞车
XAI_NOISE = re.compile(
    r'survey|review|systematic review|deep learning|neural network|'
    r'image classification|sentiment|recommendation|medical imaging|'
    r'diagnos|segmentation|tumor|cancer|radiolog', re.I)


def get(url, timeout=40, retries=2):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode('utf-8', 'ignore')
        except Exception:                                        # noqa: BLE001
            time.sleep(1.5 * (a + 1))
    return ''


def strip_html(s):
    s = re.sub(r'<[^>]+>', ' ', s or '')
    return re.sub(r'\s+', ' ', s).strip()


def crossref(q, rows=10):
    url = ('https://api.crossref.org/works?'
           + urllib.parse.urlencode({'query.bibliographic': q, 'rows': rows,
                                     'select': 'DOI,title,container-title,issued,type,abstract,author'},
                                    quote_via=urllib.parse.quote))
    raw = get(url)
    if not raw:
        return []
    try:
        items = json.loads(raw)['message']['items']
    except Exception:                                            # noqa: BLE001
        return []
    out = []
    for it in items:
        dp = (it.get('issued') or {}).get('date-parts') or [[None]]
        au = it.get('author', [])
        out.append({
            'doi': it.get('DOI', ''),
            'title': strip_html((it.get('title') or [''])[0]),
            'venue': (it.get('container-title') or [''])[0],
            'year': dp[0][0], 'type': it.get('type', ''),
            'authors': ['%s %s' % (a.get('family', ''), a.get('given', '')) for a in au[:5]],
            'abstract': strip_html(it.get('abstract', ''))[:1500],
        })
    return out


def arxiv(q, rows=10):
    url = ('http://export.arxiv.org/api/query?'
           + urllib.parse.urlencode({'search_query': q, 'max_results': rows},
                                    quote_via=urllib.parse.quote))
    raw = get(url)
    if not raw:
        return []
    x = raw
    ids = re.findall(r'<id>http://arxiv\.org/abs/([^<]+)</id>', x)
    titles = re.findall(r'<title>(.*?)</title>', x, re.S)[1:]
    sums = re.findall(r'<summary>(.*?)</summary>', x, re.S)
    pubs = re.findall(r'<published>(.*?)</published>', x)
    out = []
    for k, aid in enumerate(ids):
        t = strip_html(titles[k]) if k < len(titles) else '?'
        y = re.search(r'(\d{4})', pubs[k]) if k < len(pubs) else None
        out.append({'arxiv_id': aid, 'title': t, 'year': y.group(1) if y else None,
                    'abstract': strip_html(sums[k])[:1500] if k < len(sums) else ''})
    return out


def triage(it):
    """三档：HIGH 真撞 / MED 可疑需读摘要 / LOW XAI 噪声"""
    t = it['title'] + ' ' + it.get('abstract', '')
    has_clin = bool(CLIN.search(it['title'] + ' ' + it.get('abstract', '')))
    has_mech = bool(MECH.search(t))
    if has_clin and has_mech and not XAI_NOISE.search(it['title']):
        return 'HIGH'
    if has_clin and has_mech:
        return 'MED'
    return 'LOW'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'xai_rehab_sweep.json'))
    ap.add_argument('--sleep', type=float, default=1.1)
    args = ap.parse_args()

    rep = {
        'purpose': '「可解释/可读性 × 康复运动学」摘要级深判 —— 补上一轮只判「不撞」就放过的面',
        'questions': {
            'Q1': '康复领域是否已建立「离散符号化运动 → 临床可解释」这条路？',
            'Q2': '有没有人用 grammar / 状态机表达康复动作？',
            'Q3': '康复领域是否已借 language / translation 隐喻描述运动？',
        },
        'triage_rule': 'HIGH = 题名+摘要同时含临床词与符号化机制词且非 XAI 大杂烩；MED = 同上但命中噪声词；LOW = 其余',
        'queries': [], 'HIGH': [], 'MED': [], 'zero_hit': [],
    }

    for label, src, q in QUERIES:
        items = (crossref(q) if src == 'crossref' else arxiv(q))
        h = m = 0
        for it in items:
            it['_query'] = label
            it['_triage'] = triage(it)
            if it['_triage'] == 'HIGH':
                h += 1
                rep['HIGH'].append(it)
                print('  ★HIGH [%s] %s' % (it.get('year'), it['title'][:88]))
                print('        %s | %s' % (it.get('venue') or it.get('arxiv_id'), it['_query']))
            elif it['_triage'] == 'MED':
                m += 1
                rep['MED'].append(it)
        rec = {'label': label, 'source': src, 'query': q, 'n': len(items), 'n_high': h, 'n_med': m}
        rep['queries'].append(rec)
        print('[%s] %-34s n=%-3d HIGH=%d MED=%d' % (src[:2].upper(), label, len(items), h, m))
        if not items:
            rep['zero_hit'].append({'label': label, 'source': src, 'query': q})
            print('        ↳ 零命中')
        time.sleep(args.sleep)

    rep['summary'] = {
        'n_queries': len(QUERIES),
        'n_high': len(rep['HIGH']),
        'n_med': len(rep['MED']),
        'n_zero_hit': len(rep['zero_hit']),
        'arxiv_zero_hit': [z['label'] for z in rep['zero_hit'] if z['source'] == 'arxiv'],
    }
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n汇总：', json.dumps(rep['summary'], ensure_ascii=False, indent=1))
    print('→', args.out)


if __name__ == '__main__':
    main()
