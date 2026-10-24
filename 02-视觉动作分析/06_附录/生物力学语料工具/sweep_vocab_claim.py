#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_vocab_claim.py — 针对**核心新颖性主张**的定向撞红扫描

为什么单独写这个工具：
    「离散运动学词表（临床关节角）仍开放」这条主张，是整个选题的立身之本，
    但它此前**只靠 B4 线程的 6 篇论文**支撑，而这 6 篇**没有一篇在做临床关节角 token 化**
    （分别是：手语 BERT、机器人潜空间、关节扭转参数化、SMPL 潜变量 VQ、动作识别掩码重建）。
    第三轮之所以意外撞到 arXiv `2607.02796`，是因为**顺手挖 PDF**碰上的 ——
    说明撞红面**从未被系统扫过**。这个工具就是补这一刀。

它与 discover.py 的分工：
    discover.py  = 按线程铺开（广度优先，找全）
    本工具        = 按**主张**定向（深度优先，证伪优先）
    只报「会不会撞」，不负责把候选编进语料三 —— 入库仍须走
    curate → select → resolve → build 那条可复现链。

纪律：
    · **宁可报「没找到」也不降级充数**；null 结果本身就是结论
    · 每次都记录**检索式**与**命中数**，让「未命中」可被复核、可被推翻
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
# 每条 = (标签, 来源, 检索串)。**分开写**是为了让「未命中」能定位到是哪条式子空的。
QUERIES = [
    # —— A. 关节角本身的离散化 / 词表 ——
    ('A1 关节角+离散符号',    'crossref', 'joint angle discretization vocabulary codebook'),
    ('A2 关节角+离散token',   'crossref', 'joint angle discrete tokens sequence'),
    ('A3 临床角度+token',     'crossref', 'clinical joint angle tokenization'),
    ('A4 运动学+codebook',    'crossref', 'kinematics codebook discrete representation'),
    ('A5 角度序列+VQ',         'crossref', 'vector quantization joint angle sequence'),

    # —— B. 「语言 / 翻译」框架直接用在关节角上 ——
    ('B1 关节角+语言模型',     'crossref', 'joint angle language model translation'),
    ('B2 运动学语言',          'crossref', 'kinematic language human motion'),
    ('B3 视觉到角度+翻译',     'crossref', 'visual to joint angle translation model'),
    ('B4 生物力学语言',        'crossref', 'biomechanical language grammar joint'),

    # —— C. arXiv 侧（CS 词汇更新快，Crossref 会漏）——
    ('C1 arXiv kinematic token',    'arxiv', 'all:"joint angle" AND all:"token"'),
    ('C2 arXiv motion tokenization', 'arxiv', 'all:"motion tokenization" AND all:"joint"'),
    ('C3 arXiv kinematic vocabulary', 'arxiv', 'all:"kinematic vocabulary"'),
    ('C4 arXiv angle quantization',  'arxiv', 'all:"angle" AND all:"vector quantization" AND all:"pose"'),
]


def fetch(url, timeout=40, retries=2):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode('utf-8', 'ignore')
        except Exception:                                        # noqa: BLE001
            time.sleep(1.5 * (a + 1))
    return ''


def crossref(q, rows=8):
    u = ('https://api.crossref.org/works?query.bibliographic=%s&rows=%d'
         '&select=DOI,title,author,issued,container-title' % (urllib.parse.quote(q), rows))
    txt = fetch(u)
    if not txt:
        return [], 'fetch_failed'
    try:
        items = json.loads(txt)['message']['items']
    except Exception:                                            # noqa: BLE001
        return [], 'parse_failed'
    out = []
    for it in items:
        out.append({
            'doi': it.get('DOI', ''),
            'title': (it.get('title') or ['?'])[0],
            'year': ((it.get('issued') or {}).get('date-parts') or [[None]])[0][0],
            'venue': (it.get('container-title') or ['?'])[0],
        })
    return out, 'ok'


def arxiv(q, rows=8):
    u = 'http://export.arxiv.org/api/query?search_query=%s&max_results=%d' % (
        urllib.parse.quote(q), rows)
    txt = fetch(u)
    if not txt:
        return [], 'fetch_failed'
    ids = re.findall(r'<id>http://arxiv\.org/abs/([^<]+)</id>', txt)
    titles = [re.sub(r'\s+', ' ', t).strip() for t in
              re.findall(r'<title>(.*?)</title>', txt, re.S)][1:]
    out = [{'arxiv_id': i, 'title': t} for i, t in zip(ids, titles)]
    return out, 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'vocab_claim_sweep.json'))
    ap.add_argument('--sleep', type=float, default=1.3)
    ap.add_argument('--only', default='', help='只跑标签前缀，如 A 或 B')
    args = ap.parse_args()

    report = {'purpose': '核心新颖性主张「离散运动学词表（临床关节角）仍开放」的定向撞红扫描',
              'note': 'null 结果也是结论：说明这些式子在 Crossref/arXiv 公开索引里取不到，'
                      '但不等于「不存在」——入库仍须走可复现链并做多源交叉证实',
              'queries': []}

    for label, src, q in QUERIES:
        if args.only and not label.startswith(args.only):
            continue
        items, status = (crossref(q) if src == 'crossref' else arxiv(q))
        print('─' * 96)
        print('【%s】%s  [%s]' % (label, q, status))
        print('  命中 %d' % len(items))
        for it in items[:6]:
            print('   · %s' % (it.get('title', '')[:88]))
            print('     %s | %s %s' % (it.get('doi') or it.get('arxiv_id'),
                                       it.get('year', ''), (it.get('venue') or '')[:34]))
        report['queries'].append({'label': label, 'source': src, 'query': q,
                                  'status': status, 'n': len(items), 'hits': items})
        time.sleep(args.sleep)

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    tot = sum(q['n'] for q in report['queries'])
    zero = [q['label'] for q in report['queries'] if q['n'] == 0]
    print('=' * 96)
    print('共 %d 条检索式，命中合计 %d；零命中：%s' % (len(report['queries']), tot, zero or '无'))
    print('已写出：%s' % args.out)


if __name__ == '__main__':
    main()
