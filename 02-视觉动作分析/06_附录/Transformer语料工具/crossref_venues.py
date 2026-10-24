#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crossref_venues.py — 用 Crossref 官方 API 反查每篇待下载论文的**真实载体**（期刊/会议）。

为什么必须查而不是凭印象写：
  「去哪找」完全取决于载体。IEEE TNSRE 要走 IEEE Xplore，MDPI/Frontiers 直接官网免费下载，
  CVPR/ICCV 走 CVF，ACL 走 Anthology，NeurIPS/ICLR 走官方 proceedings，Elsevier 走 ScienceDirect。
  凭记忆写载体就是把猜测包装成事实——本项目的交付规范明确禁止。

输出：
  crossref_venues.json  每条的 DOI / 期刊名 / 出版商 / 年份 / 类型 / 匹配分 / 建议数据库
  只采信匹配分 ≥ 阈值的记录；低于阈值标 UNVERIFIED，不写入载体。

用法：python3 crossref_venues.py
"""
import difflib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = 'kb-venue-lookup/1.0 (mailto:research@example.org)'   # Crossref polite pool
API = 'https://api.crossref.org/works'
THRESH = 0.72

# 出版商 → 该去哪个数据库 / 是否开放获取
ROUTE = [
    (r'ieee',            'IEEE Xplore（需订阅）',                        'IEEE'),
    (r'mdpi',            'MDPI 官网（开放获取，直接下载）',              'MDPI'),
    (r'frontiers',       'Frontiers 官网（开放获取，直接下载）',          'Frontiers'),
    (r'springer',        'SpringerLink（需订阅）',                      'Springer'),
    (r'elsevier|sciencedirect', 'ScienceDirect（需订阅，部分 OA）',      'Elsevier'),
    (r'wiley',           'Wiley Online Library（需订阅，部分 OA）',      'Wiley'),
    (r'acm',             'ACM Digital Library（需订阅）',                'ACM'),
    (r'\bAAAI\b',        'AAAI Proceedings（开放获取）',                 'AAAI'),
    (r'\bEMNLP\b|\bACL\b', 'ACL Anthology（开放获取）',                  'ACL'),
    (r'\bCVPR\b|\bICCV\b|\bECCV\b', 'CVF Open Access（开放获取）',      'CVF'),
    (r'\bNeurIPS\b',     'NeurIPS Proceedings（开放获取）',              'NeurIPS'),
    (r'\bICLR\b',        'OpenReview（开放获取）',                      'OpenReview'),
    (r'\bICML\b|\bPMLR\b', 'PMLR（开放获取）',                          'PMLR'),
    (r'\bACM MM\b|\bMM\b', 'ACM DL（需订阅）',                          'ACM'),
]


def norm(s):
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).split())


def route_for(container: str, publisher: str, type_: str) -> tuple[str, str]:
    blob = '%s %s %s' % (container or '', publisher or '', type_ or '')
    for pat, db, kind in ROUTE:
        if re.search(pat, blob, re.I):
            return db, kind
    return '**需按题名确认载体后再定位数据库**', '未知'


def query(title):
    url = '%s?query.bibliographic=%s&rows=5' % (API, urllib.parse.quote(title))
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/json'})
    data = json.loads(urllib.request.urlopen(req, timeout=45).read().decode('utf-8'))
    out = []
    for it in data.get('message', {}).get('items', []):
        t = (it.get('title') or [''])[0]
        if not t:
            continue
        cts = it.get('container-title') or []
        out.append({
            'title': t,
            'container': cts[0] if cts else '',
            'short_container': (it.get('short-container-title') or [''])[0],
            'publisher': it.get('publisher', ''),
            'doi': it.get('DOI', ''),
            'type': it.get('type', ''),
            'year': (it.get('issued', {}).get('date-parts') or [[None]])[0][0],
            'event': (it.get('event') or {}).get('name', ''),
            'page': it.get('page', ''),
            'volume': it.get('volume', ''),
            'score': round(difflib.SequenceMatcher(None, norm(title), norm(t)).ratio(), 3),
        })
    out.sort(key=lambda x: -x['score'])
    return out


def main():
    res_path = HERE / 'corpus_resolved.json'
    cand_path = HERE / 'corpus_candidates.json'
    cand = json.loads(cand_path.read_text(encoding='utf-8'))
    byt = {p['title']: p for p in cand['papers']}
    res = json.loads(res_path.read_text(encoding='utf-8'))
    todo = [p for p in res['papers'] if p.get('status') != 'confirmed']

    results = {}
    for n, p in enumerate(todo, 1):
        title = p['title']
        have_doi = (byt.get(title) or {}).get('doi', '')
        try:
            hits = query(title)
        except Exception as e:                       # noqa: BLE001
            print('  ERR  %-52s %r' % (title[:50], e), flush=True)
            time.sleep(2); continue
        top = hits[0] if hits else None
        ok = bool(top and top['score'] >= THRESH)
        container = top['container'] or top['event'] if ok else ''
        db, kind = route_for(container, top['publisher'] if ok else '', top['type'] if ok else '')
        results[title] = {
            'verified': ok,
            'score': top['score'] if top else 0.0,
            'venue': container,
            'event': top['event'] if top else '',
            'publisher': top['publisher'] if top else '',
            'doi': (top['doi'] if top else '') or have_doi,
            'year': top['year'] if top else None,
            'type': top['type'] if top else '',
            'matched_title': top['title'] if top else '',
            'database': db,
            'oa_kind': kind,
            'alternatives': [{'title': h['title'], 'container': h['container'] or h['event'],
                              'doi': h['doi'], 'score': h['score']} for h in hits[1:3]],
        }
        flag = '✓' if ok else '✗'
        print('%s %2d/%d %.2f  %-46s -> %s' % (flag, n, len(todo),
              results[title]['score'], title[:44], (container or '未核实')[:44]), flush=True)
        time.sleep(1.2)

    (HERE / 'crossref_venues.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    v = sum(1 for r in results.values() if r['verified'])
    print('\n核实 %d / %d（阈值 %.2f）→ crossref_venues.json' % (v, len(todo), THRESH))


if __name__ == '__main__':
    main()
