#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_clinical.py — **临床/康复侧**定向撞红扫描（补系统性盲区）

为什么必须单独写这个工具：
    第十八轮做完第二代引文追溯后，我统计了一下已坐实的六条撞车：
    F5 DigitCode / F6 pose grammar / F7 VQ-HPS / F8 Hourglass Tokenizer /
    F9 PAHMT / F10 pose grammar 后继 —— **六条全部在 CV 侧**。
    换句话说，我从头到尾**只查了「CV 有没有人做过」，从来没查过
    「康复/临床侧有没有人做过」**。

    而 Pohl 等 2026（JMIR RAT，已入库，全文级）恰好从**临床侧**给出了反直觉的结论：
    临床医师判断动作质量只看 smoothness / efficiency / compensation，
    **「joint angle」全文 0 次**，且**不熟悉 jerk 与 interjoint coordination**。
    —— 这说明临床侧不仅是个独立的撞红面，而且**它的偏好和我们的技术直觉不一样**。

    这个工具就是补这一刀。风险面具体有五类：
      A. 记谱法（**Labanotation 等）进康复** —— DigitCode(F5) 用的正是 Hand Labanotation，
         而 Labanotation 在物理治疗/舞蹈治疗里**本身就有真实传统**，这是最可能的临床侧撞车
      B. 康复技能/动作的**符号化或语法化**表示
      C. **digital biomarkers / digital measures** 上肢（临床侧自己的指标体系）
      D. 康复领域**借「语言 / 翻译 / 语言模型」做隐喻**
      E. 面向临床的**可解释运动学 / 可解释 AI**

与既有工具的分工：
    discover.py         = 按线程铺开（广度优先）
    sweep_vocab_claim.py= 按**主张**在 CV 侧证伪（深度优先）
    sweep_intersection.py = 跨线程交集验证
    本工具              = 按**领域**（临床/康复侧）证伪 —— **补的是前三个都没覆盖的那一面**

纪律：
    · **宁可报「没找到」也不降级充数**；零命中本身就是要留证的结论
    · 记录**每条检索式**与**命中数**，让「未命中」可被复核、可被推翻
    · **Crossref 零命中不可信**（同形异义污染，见 K2），只有 arXiv 零命中可作空白证据
    · 不绕过反爬；只走 Crossref / arXiv 公开 API
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

# ---------------------------------------------------------------- 检索式
# 每条 = (标签, 来源, 检索串)。分开写是为了让「未命中」能定位到**是哪条式子空的**。
QUERIES = [
    # —— A. 记谱法进康复：最可能的临床侧撞车（DigitCode 的字母表来源）——
    ('A1 Labanotation+康复',      'crossref', 'Labanotation rehabilitation upper limb'),
    ('A2 Labanotation+物理治疗',   'crossref', 'Labanotation physical therapy movement analysis'),
    ('A3 Labanotation+卒中',      'crossref', 'Labanotation stroke rehabilitation notation'),
    ('A4 记谱法+上肢',            'crossref', 'movement notation system upper limb clinical'),
    ('A5 记谱法+临床评估',        'crossref', 'dance notation clinical movement assessment'),
    ('A6 arXiv Labanotation 康复',  'arxiv', 'all:"Labanotation" AND (all:"rehabilitation" OR all:"stroke")'),
    ('A7 arXiv Labanotation 上肢',  'arxiv', 'all:"Labanotation" AND all:"upper limb"'),
    ('A8 arXiv motion notation 康复','arxiv', 'all:"motion notation" AND all:"rehabilitation"'),

    # —— B. 康复技能/动作的符号化、语法化表示 ——
    ('B1 康复+符号化动作',        'crossref', 'symbolic representation movement rehabilitation therapy'),
    ('B2 上肢+语法+技能',         'crossref', 'grammar skill representation upper limb rehabilitation'),
    ('B3 离散码本+康复动作',      'crossref', 'discrete codebook rehabilitation movement representation'),
    ('B4 arXiv 康复 token 动作',   'arxiv', 'all:"rehabilitation" AND all:"token" AND all:"movement representation"'),

    # —— C. digital biomarkers / digital measures 上肢（临床自己的指标体系）——
    ('C1 上肢+数字生物标志物',    'crossref', 'digital biomarkers upper limb neurological rehabilitation'),
    ('C2 digital measures 上肢',  'crossref', 'digital measures of functioning upper limb rehabilitation'),
    ('C3 数字指标+运动学+临床',    'crossref', 'digital kinematic metrics clinical interpretation rehabilitation'),
    ('C4 数字表型+卒中上肢',      'crossref', 'digital phenotype upper limb stroke quantitative'),

    # —— D. 康复领域借「语言/翻译」做隐喻 ——
    ('D1 康复+语言模型',          'crossref', 'language model rehabilitation movement assessment'),
    ('D2 运动+机器翻译',          'crossref', 'machine translation human movement clinical'),
    ('D3 康复+自然语言+运动',     'crossref', 'natural language movement rehabilitation physiotherapy'),
    ('D4 arXiv 康复 翻译 动作',    'arxiv', 'all:"rehabilitation" AND all:"translation" AND all:"joint angle"'),

    # —— E. 面向临床的可解释运动学 / 可解释 AI ——
    ('E1 可解释+运动学+临床',      'crossref', 'interpretable kinematics clinical decision support'),
    ('E2 可解释AI+康复评估',       'crossref', 'explainable AI rehabilitation assessment clinicians'),
    ('E3 物理治疗+可解释+模型',     'crossref', 'physiotherapy interpretable model patient interpretation'),
    ('E4 arXiv 可解释 上肢 康复',   'arxiv', 'all:"interpretable" AND all:"upper limb" AND all:"rehabilitation"'),
]

# 临床侧「已知钉子」：命中即重点复核
NEEDLES = re.compile(
    r'labanotation|easley|kinetic|notation|digital biomarker|digital measure|'
    r'digital phenotyp|interpretab|explainab|clinician|clinical decision|'
    r'token|vocabulary|codebook|grammar|language model|translation',
    re.I)


def fetch(url, timeout=40, retries=2):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception:
            if a + 1 == retries:
                return b''
            time.sleep(1.5)
    return b''


def crossref(q, rows=8):
    url = ('https://api.crossref.org/works?'
           + urllib.parse.urlencode({'query.bibliographic': q, 'rows': rows,
                                     'select': 'DOI,title,container-title,issued,type'},
                                    quote_via=urllib.parse.quote))
    raw = fetch(url)
    if not raw:
        return []
    try:
        items = json.loads(raw)['message']['items']
    except Exception:
        return []
    out = []
    for it in items:
        dp = (it.get('issued') or {}).get('date-parts') or [[None]]
        out.append({
            'doi': it.get('DOI', ''),
            'title': (it.get('title') or [''])[0],
            'venue': (it.get('container-title') or [''])[0],
            'year': dp[0][0],
            'type': it.get('type', ''),
        })
    return out


def arxiv(q, rows=8):
    url = ('http://export.arxiv.org/api/query?'
           + urllib.parse.urlencode({'search_query': q, 'max_results': rows},
                                    quote_via=urllib.parse.quote))
    raw = fetch(url)
    if not raw:
        return []
    x = raw.decode('utf-8', 'replace')
    ids = re.findall(r'<id>http://arxiv\.org/abs/([^<]+)</id>', x)
    titles = re.findall(r'<title>(.*?)</title>', x, re.S)[1:]
    pubs = re.findall(r'<published>(.*?)</published>', x)
    out = []
    for aid, t in zip(ids, titles):
        t = re.sub(r'\s+', ' ', t).strip()
        y = re.search(r'(\d{4})', pubs[ids.index(aid)]) if aid in pubs else None
        out.append({'arxiv_id': aid, 'title': t, 'year': y.group(1) if y else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'clinical_sweep.json'))
    ap.add_argument('--sleep', type=float, default=1.2)
    args = ap.parse_args()

    report = {
        'purpose': '临床/康复侧定向撞红扫描 —— 补「只查了 CV 侧」的系统性盲区',
        'discipline': [
            'Crossref 零命中不可信（同形异义污染，教训 K2）；只有 arXiv 零命中可作空白证据',
            '不绕过反爬；只走 Crossref / arXiv 公开 API',
            '本工具只报「会不会撞」，不入库 —— 入库仍须走 curate→select→resolve→build',
        ],
        'queries': [], 'hits': [], 'zero_hit_queries': [],
    }

    for label, src, q in QUERIES:
        items = (crossref(q) if src == 'crossref' else arxiv(q))
        key = 'doi' if src == 'crossref' else 'arxiv_id'
        flagged = [it for it in items if NEEDLES.search(it.get('title', ''))]
        rec = {'label': label, 'source': src, 'query': q,
               'n': len(items), 'n_flagged': len(flagged)}
        report['queries'].append(rec)
        print('[%s] %-28s n=%d  标记 %d' % (src[:2].upper(), label, len(items), len(flagged)))
        if not items:
            report['zero_hit_queries'].append({'label': label, 'source': src, 'query': q})
        for it in flagged:
            it['_query'] = label
            report['hits'].append(it)
            print('        ↳ [%s] %s' % (it.get('year'), it['title'][:92]))
        time.sleep(args.sleep)

    report['summary'] = {
        'n_queries': len(QUERIES),
        'n_zero_hit': len(report['zero_hit_queries']),
        'n_flagged_unique': len({(h.get('doi') or h.get('arxiv_id')) for h in report['hits']}),
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n汇总：', json.dumps(report['summary'], ensure_ascii=False))
    print('零命中检索式：', len(report['zero_hit_queries']))
    print('→', args.out)


if __name__ == '__main__':
    main()
