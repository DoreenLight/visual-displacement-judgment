#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_middle.py — 「多媒体 / 语义运动表示」**中间地带**广度扫描（V9h）

为什么必须补这一刀：
    前几轮的检索面一直只有**两面**：
        康复/临床侧  → sweep_clinical.py
        CV/3D 姿态侧  → discover.py / sweep_vocab_claim.py
    **中间那块没人查过。** 而 F15（ICMCS 2012,
    *An Ontology for video human movement representation based on Benesh notation*）
    恰恰就掉在中间 —— 它既不是临床研究，也不是 3D 姿态估计，
    **却在 37 条 Benesh 年表里被我偶然撞见**。
    这正是 V9e 老教训的第三次复现：**式子检索会漏，而且漏的是你没想过要去查的那一块。**

    这个地带通常包括：human movement description（HMD）、运动本体（ontology）、
    动作/手势描述语言、运动捕捉数据标准与交换格式、定性运动表示、
    动作 tokenization 等等。**它是「离散符号化运动表示」这条线的古典老家**，
    而我们恰恰要在这条线上做东西 —— **不查它，等于不打地基就盖楼。**

分带（这是本工具的核心）：
    命中的每条都要归到**一个带**里，好知道自己在哪：
      [中间带] = 语义/本体/描述语言/数据标准/定性表示  ← **真正要查的**
      [CV]     = 3D 姿态估计 / markerless / 视频重建
      [临床]    = 康复 / 卒中 / 临床评估
      [艺术]    = 舞蹈 / 表演 / 记谱

纪律：
    · Crossref 零命中不可信（K2 同形异义）；**arXiv 零命中可作空白证据**
    · **作者姓氏会污染关键词检索**（K9）→ 可疑条目一律回查作者字段
    · 标题级 ≠ 结论级；关键条目要取摘要
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
    # —— 人类运动描述（HMD）：这个领域的旗舰术语 ——
    ('M1 human movement description',   'crossref', 'human movement description multimedia representation'),
    ('M2 HMD 动作描述',                 'crossref', 'human movement description standard annotation format'),
    ('M3 arXiv HMD',                    'arxiv',   'all:"human movement description"'),

    # —— 运动本体 / 语义表示 ——
    ('M4 运动本体',                     'crossref', 'ontology human movement video semantic representation'),
    ('M5 语义运动表示',                 'crossref', 'semantic representation human motion understanding video'),
    ('M6 arXiv 运动本体',               'arxiv',   'all:"ontology" AND all:"human movement"'),

    # —— 描述语言 / 定性表示 ——
    ('M7 运动描述语言',                 'crossref', 'motion description language human movement specification'),
    ('M8 定性运动表示',                 'crossref', 'qualitative motion representation human gesture description'),
    ('M9 手势描述语言',                 'crossref', 'gesture description language formal representation'),
    ('M10 arXiv 定性运动表示',          'arxiv',   'all:"qualitative" AND all:"motion representation" AND all:"human"'),

    # —— 动作 tokenization / 离散动作表示（非姿态估计）——
    ('M11 离散动作表示',                'crossref', 'discrete action representation tokenization video understanding'),
    ('M12 arXiv 动作 token',            'arxiv',   'all:"action tokens" AND all:"video"'),

    # —— 数据标准 / 交换格式（符号化运动的「古典老家」）——
    ('M13 动捕数据标准',                'crossref', 'motion capture data standard interchange format human movement'),
    ('M14 BVH 动捕格式语义',            'crossref', 'BVH motion capture format semantic annotation skeleton'),
    ('M15 arXiv 动捕标准',              'arxiv',   'all:"motion capture" AND all:"interchange format"'),

    # —— 数字人 / 动画标准里的运动描述 ——
    ('M16 MPEG 人体动画',               'crossref', 'MPEG-4 human body animation parametric representation standard'),
    ('M17 arXiv 数字人运动表示',        'arxiv',   'all:"digital human" AND all:"motion representation" AND all:"standard"'),
]

# ---- 分带判据 ----
Z_MIDDLE = re.compile(
    r'human movement description|movement description|description language|'
    r'ontolog|semantic representation|qualitative motion|motion description|'
    r'interchange format|data standard|parametric representation|'
    r'gesture description|formal representation|annotation format', re.I)
Z_CV = re.compile(
    r'3d (human )?pose estimat|pose estimat|markerless|monocular|multi-view|'
    r'keypoint|human mesh|smpl|video-based human|reconstruct', re.I)
Z_CLIN = re.compile(
    r'rehabil|clinical|clinician|patient|stroke|therap|neurorehab|'
    r'physiother|occupational therap|paresis|activities of daily', re.I)
Z_ART = re.compile(
    r'laban|benesh|dance|choreograph|notat|ballet|arts|performance|entrain', re.I)


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
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', s or '')).strip()


def crossref(q, rows=12):
    url = ('https://api.crossref.org/works?'
           + urllib.parse.urlencode({'query.bibliographic': q, 'rows': rows,
                                     'select': 'DOI,title,container-title,issued,type,author,abstract'},
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
            'id': it.get('DOI', ''), 'title': strip_html((it.get('title') or [''])[0]),
            'venue': (it.get('container-title') or [''])[0], 'year': dp[0][0],
            'type': it.get('type', ''),
            'authors': ['%s %s' % (a.get('family', ''), a.get('given', '')) for a in au[:5]],
            'n_auth': len(au), 'abstract': strip_html(it.get('abstract', ''))[:1200],
        })
    return out


def arxiv(q, rows=12):
    url = ('http://export.arxiv.org/api/query?'
           + urllib.parse.urlencode({'search_query': q, 'max_results': rows},
                                    quote_via=urllib.parse.quote))
    raw = get(url)
    if not raw:
        return []
    ids = re.findall(r'<id>http://arxiv\.org/abs/([^<]+)</id>', raw)
    titles = re.findall(r'<title>(.*?)</title>', raw, re.S)[1:]
    sums = re.findall(r'<summary>(.*?)</summary>', raw, re.S)
    pubs = re.findall(r'<published>(.*?)</published>', raw)
    out = []
    for k, aid in enumerate(ids):
        y = re.search(r'(\d{4})', pubs[k]) if k < len(pubs) else None
        out.append({'id': aid, 'title': strip_html(titles[k]) if k < len(titles) else '?',
                    'venue': 'arXiv', 'year': y.group(1) if y else None, 'type': 'preprint',
                    'authors': [], 'n_auth': 0,
                    'abstract': strip_html(sums[k])[:1200] if k < len(sums) else ''})
    return out


def zone(it):
    """归带。中间带优先（它是本工具真正要找的），其次 CV / 临床 / 艺术。"""
    t = it['title'] + ' ' + it.get('abstract', '')
    if Z_MIDDLE.search(t):
        return '中间带'
    if Z_CV.search(t):
        return 'CV'
    if Z_CLIN.search(t):
        return '临床'
    if Z_ART.search(t):
        return '艺术'
    return '其他'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'middle_sweep.json'))
    ap.add_argument('--sleep', type=float, default=1.0)
    args = ap.parse_args()

    rep = {
        'purpose': '「多媒体/语义运动表示」中间地带广度扫描（V9h）—— 补前几轮从未查过的第三面',
        'why': 'F15（ICMCS 2012 Benesh video ontology）证明中间地带有货，而前两轮只打了康复面与 CV 面',
        'zones': {'中间带': '语义/本体/描述语言/数据标准/定性表示 ← 真正要查的',
                  'CV': '3D 姿态估计 / markerless / 视频重建',
                  '临床': '康复 / 卒中 / 临床评估',
                  '艺术': '舞蹈 / 表演 / 记谱'},
        'queries': [], 'by_zone': {}, 'zero_hit': [], 'seen': {},
    }

    for label, src, q in QUERIES:
        items = (crossref(q) if src == 'crossref' else arxiv(q))
        zc = {}
        for it in items:
            it['_query'] = label
            z = zone(it)
            it['_zone'] = z
            zc[z] = zc.get(z, 0) + 1
            if it['id']:
                rep['seen'][it['id']] = it
            if z == '中间带':
                rep['by_zone'].setdefault('中间带', []).append(it)
                print('  ◆[%s] %s' % (it.get('year'), it['title'][:88]))
                print('     %s | %s' % (it.get('venue', '')[:52], it['_query']))
        rep['queries'].append({'label': label, 'source': src, 'query': q,
                               'n': len(items), 'zones': zc})
        print('[%s] %-32s n=%-3d %s' % (src[:2].upper(), label, len(items),
                                        {k: v for k, v in zc.items() if k != '其他'} or '(全落其他)'))
        if not items:
            rep['zero_hit'].append({'label': label, 'source': src, 'query': q})
            print('        ↳ 零命中')
        time.sleep(args.sleep)

    seen_zone = {}
    for i in rep['seen'].values():
        seen_zone[i['_zone']] = seen_zone.get(i['_zone'], 0) + 1
    rep['summary'] = {
        'n_queries': len(QUERIES),
        'n_unique': len(rep['seen']),
        'zone_dist': seen_zone,
        'n_middle_unique': len({i['id'] for i in rep['by_zone'].get('中间带', []) if i['id']}),
        'arxiv_zero_hit': [z['label'] for z in rep['zero_hit'] if z['source'] == 'arxiv'],
    }
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n汇总：', json.dumps(rep['summary'], ensure_ascii=False, indent=1))
    print('→', args.out)


if __name__ == '__main__':
    main()
