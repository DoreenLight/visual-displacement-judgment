#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_intersection.py — 第二阶段：测「交集真空」这个断言本身

背景：
    第一阶段（sweep_vocab_claim.py）撞到 4 篇先行工作
    （2402.17192 / TMRB 2026 / 2607.02796 / 2608.03127），
    归纳出一个推断：**它们都撞在"生物力学/运动学"侧，没有一篇撞在
    "离散符号化"这个机制本身；交集（上肢 + 视觉直接预测 + 生物力学 grammar + 康复）看起来是空的。**

    但那 4 篇**都是"顺手碰上的"**，不是系统检索的产物。
    **用 4 个偶遇样本去断言一个交集是空的，方法上不成立。**
    这个工具就是来把这句话证伪（或证实）的。

方法上的三条硬规矩（都是上一轮踩出来的）：
  1. **arXiv 全字段检索（all: AND all:）是唯一可信的零命中来源。**
     Crossref 的 `query.bibliographic` 会被同形异义污染
     （"visual to joint angle" → "Visual Angle" 感知科学；"…language grammar" → 语言学专著），
     所以本工具 Crossref 侧只作**补充线索**，不作空白证据。
  2. **负结果必须写成「这些式子取不到」而不是「该方向不存在」。**
  3. **不绕过反爬**；只走 Crossref / arXiv 公开 API。

只回答一个问题：**有没有人做过【上肢/腕肘肩】+【视觉或单目输入】+【离散/符号化表征】？**
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

# 三段式交集式子：{部位} × {输入} × {离散机制}
UPPER = 'upper limb OR upper-limb OR wrist OR elbow OR shoulder OR forearm'
INPUT = 'monocular OR "single camera" OR RGB-D OR "markerless" OR "video-based" OR visual'
DISCRETE = ('token OR tokenization OR tokenisation OR discrete OR symbolic OR vocabulary '
            'OR "codebook" OR "vector quantization" OR VQ')

QUERIES = [
    # —— 交集直击：三段全含 ——
    ('X1 上肢∩视觉∩离散',   'arxiv', f'all:{UPPER} AND all:{INPUT} AND all:{DISCRETE}'),
    ('X2 腕/前臂∩视觉∩离散', 'arxiv', 'all:wrist AND all:monocular AND all:discrete'),
    ('X3 关节角∩视觉∩token',  'arxiv', 'all:"joint angle" AND all:monocular AND all:token'),
    ('X4 关节角∩视觉∩离散',  'arxiv', 'all:"joint angles" AND all:video AND all:discrete'),

    # —— 两段：离散机制 × 上肢/腕（不限输入），找"有人做了符号化但部位不同" ——
    ('Y1 离散符号化∩上肢',   'arxiv', f'all:{UPPER} AND all:{DISCRETE}'),
    ('Y2 词表∩关节角',       'arxiv', 'all:vocabulary AND all:"joint angle"'),
    ('Y3 codebook∩运动学',   'arxiv', 'all:codebook AND all:kinematic'),

    # —— 两段：生物力学/临床 × 离散机制（不限部位），找"有人做了符号化但非视觉" ——
    ('Z1 生物力学∩离散',     'arxiv', 'all:biomechanic* AND all:' + DISCRETE),
    ('Z2 康复∩离散',        'arxiv', 'all:rehabilitation AND all:' + DISCRETE),
    ('Z3 临床关节角∩token',  'arxiv', 'all:"clinical" AND all:"joint angle" AND all:token'),

    # —— 词表这个词本身在运动学里到底被怎么用 ——
    ('W1 motion vocabulary',  'arxiv', 'all:"motion vocabulary"'),
    ('W2 kinematic token',    'arxiv', 'all:"kinematic token"'),
    ('W3 joint vocabulary',   'arxiv', 'all:"joint vocabulary"'),
]


def fetch(url, timeout=45, retries=2):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode('utf-8', 'ignore')
        except Exception:                                        # noqa: BLE001
            time.sleep(2.0 * (a + 1))
    return ''


def arxiv(q, rows=12):
    u = 'http://export.arxiv.org/api/query?search_query=%s&max_results=%d' % (
        urllib.parse.quote(q), rows)
    txt = fetch(u)
    if not txt:
        return [], 'fetch_failed'
    ids = re.findall(r'<id>http://arxiv\.org/abs/([^<]+)</id>', txt)
    titles = [re.sub(r'\s+', ' ', t).strip() for t in
              re.findall(r'<title>(.*?)</title>', txt, re.S)][1:]
    return ([{'arxiv_id': i, 'title': t} for i, t in zip(ids, titles)], 'ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'intersection_sweep.json'))
    ap.add_argument('--sleep', type=float, default=1.4)
    ap.add_argument('--only', default='', help='只跑标签前缀 X/Y/Z/W')
    args = ap.parse_args()

    report = {
        'purpose': '第二阶段：证伪/证实「上肢∩视觉直接预测∩离散符号化∩生物力学 grammar 仍为空」',
        'method_note': 'arXiv 全字段检索的零命中可作空白证据；Crossref 侧被同形异义污染，不作空白证据。'
                       'null 结果 = 这些式子取不到，不等于该方向不存在。',
        'queries': [],
    }

    for label, src, q in QUERIES:
        if args.only and not label.startswith(args.only):
            continue
        items, status = arxiv(q)
        print('─' * 98)
        print('【%s】%s  [%s]  命中 %d' % (label, src, status, len(items)))
        for it in items[:8]:
            print('   · %-70s %s' % (it['title'][:70], it['arxiv_id']))
        report['queries'].append({'label': label, 'query': q, 'status': status,
                                  'n': len(items), 'hits': items})
        time.sleep(args.sleep)

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    zero = [q['label'] for q in report['queries'] if q['n'] == 0]
    nonzero = [(q['label'], q['n']) for q in report['queries'] if q['n'] > 0]
    print('=' * 98)
    print('零命中（可作空白一侧的证据）: %s' % (zero or '无'))
    print('有命中（须逐条判是否撞）      : %s' % (nonzero or '无'))
    print('已写出：%s' % args.out)


if __name__ == '__main__':
    main()
