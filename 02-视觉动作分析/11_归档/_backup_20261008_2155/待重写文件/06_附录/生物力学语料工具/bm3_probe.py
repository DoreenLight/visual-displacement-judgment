#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语料三待取件：并发探测 Europe PMC / 校验官方 OA 直链，产出 bm3_missing.json。

只读 01_参考论文/生物力学线/index.json + 磁盘；只写自己的报告文件。
不写 biomech_download_report.json / biomech_recovery_report.json。
"""
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BM = ROOT / '01_参考论文' / '生物力学线'
OUT = HERE / 'bm3_missing.json'
EPMC = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36'
BLOCK = ('bm-verify', 'Just a moment', 'Checking your browser', 'Access to this page has been denied',
         'cf-browser-verification', 'Enable JavaScript and cookies', 'Attention Required!',
         'We apologize for the inconvenience')


def get(url, timeout=35):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': '*/*'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(), r.status, r.geturl()
    except Exception as e:                                        # noqa: BLE001
        return b'', 0, type(e).__name__


def norm(s):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', s or '').lower())


def thread_dir(thread):
    for d in BM.iterdir():
        if d.is_dir() and d.name.startswith(thread + '_'):
            return d
    return None


def on_disk(td, title):
    if not td:
        return None
    k = norm(title)[:38]
    for f in list(td.glob('*.pdf')) + list(td.glob('*.xml')):
        if k in norm(f.stem) or norm(f.stem)[:38] in k:
            return f
    return None


def probe(p):
    doi = (p.get('doi') or '').strip()
    r = {'epmc': 'no_doi'}
    if doi:
        url = '%s?query=%s&format=json&pageSize=2&resultType=core' % (EPMC, urllib.parse.quote('DOI:"%s"' % doi))
        body, st, _ = get(url, timeout=30)
        if st == 200 and body:
            try:
                res = json.loads(body.decode('utf-8', 'ignore')).get('resultList', {}).get('result', [])
            except Exception:                                    # noqa: BLE001
                res = []
            hit = next((x for x in res if (x.get('doi') or '').lower() == doi.lower()), None)
            r = ({'epmc': 'hit', 'pmcid': hit.get('pmcid', ''), 'isOA': hit.get('isOpenAccess', ''),
                  'inEPMC': hit.get('inEPMC', '')} if hit else {'epmc': 'no_record'})
        else:
            r = {'epmc': 'query_failed'}
    # 官方 OA 直链可达性（只探 HEAD 式的 GET，不落地文件）
    for key in ('oa_pdf', 'url_pdf'):
        u = (p.get(key) or '').strip()
        if not u:
            continue
        body, st, final = get(u, timeout=40)
        if not body:
            continue
        head = body[:4000]
        if body.startswith(b'%PDF'):
            r[key] = {'state': 'ok', 'bytes': len(body)}
            break
        txt = body.decode('utf-8', 'ignore')
        if any(m in txt for m in BLOCK):
            r[key] = {'state': 'blocked', 'why': next(m for m in BLOCK if m in txt)}
        else:
            m = re.findall(r'href="([^"]+\.pdf[^"]*)"', txt, re.I)
            r[key] = {'state': 'landing_page', 'pdf_links': len(m), 'final': final[:120]}
    return r


def main():
    idx = json.loads((BM / 'index.json').read_text(encoding='utf-8'))
    todo = []
    for p in idx['papers']:
        if not (p.get('title') or '').strip():
            continue
        if on_disk(thread_dir(p['thread']), p['title']):
            continue
        todo.append(p)
    print('待取件 %d 篇，开始并发探测…' % len(todo), flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        probes = list(ex.map(probe, todo))
    out = []
    for p, pr in zip(todo, probes):
        out.append({**{k: p.get(k, '') for k in
                       ('title', 'thread', 'thread_name', 'lens_name', 'year', 'venue',
                        'doi', 'arxiv_id', 'is_oa', 'oa_status', 'evidence_grade',
                        'oa_landing', 'url_abs', 'relevance', 'note')}, **pr})
    OUT.write_text(json.dumps({'generated': 'bm3_probe.py', 'items': out}, ensure_ascii=False, indent=1),
                   encoding='utf-8')
    from collections import Counter
    print('epmc:', dict(Counter(i['epmc'] for i in out)))
    print('isOA:', dict(Counter(i.get('isOA', '-') for i in out)))
    print('直链:', dict(Counter((i.get('oa_pdf') or {}).get('state', '-') for i in out)))
    print('报告:', OUT.relative_to(ROOT))


if __name__ == '__main__':
    main()
