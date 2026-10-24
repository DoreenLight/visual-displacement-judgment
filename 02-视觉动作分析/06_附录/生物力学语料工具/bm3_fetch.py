#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语料三（生物力学与上肢运动学）隔离版取件器。

与并发工作流里 recover_oa.py / download_oa.py 的隔离点：
  - 只读 01_参考论文/整合系统/生物力学线/index.json 作为候选源
  - 产物只写 01_参考论文/整合系统/生物力学线/<线程>/ 下的 .pdf / .xml
  - 报告只写 06_附录/生物力学语料工具/bm3_fetch_report.json
    **不写** biomech_download_report.json / biomech_recovery_report.json

取件通道（全部为出版商或官方仓储的合法开放位置）：
  1. Europe PMC 官方 REST  fullTextXML   —— 生物医学类官方 OA 镜像，实测可拿到完整 JATS 正文
  2. Unpaywall 已记录的 oa_pdf 直链        —— 出版商/机构仓储官方直链
  3. oa_landing 直链探测                   —— 仅当直链本身以 .pdf 结尾
  4. arXiv（若条目有 arxiv_id）

纪律：不使用影子图书馆，不规避付费墙，不绕过反爬验证。
反爬拦截的条目如实标记 blocked，交人工浏览器处理。
"""
import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BM = ROOT / '01_参考论文' / '整合系统/生物力学线'
REPORT = HERE / 'bm3_fetch_report.json'

EPMC = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
EPMC_XML = 'https://www.ebi.ac.uk/europepmc/webservices/rest/%s/fullTextXML'
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36'

BLOCK_MARKERS = (
    'bm-verify', 'captcha', 'Just a moment', 'Checking your browser',
    'Access to this page has been denied', 'cf-browser-verification',
    'Enable JavaScript and cookies to continue', 'Attention Required!',
)


def get(url, timeout=60, retries=2):
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': '*/*'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), r.status, r.geturl()
        except Exception as e:                                    # noqa: BLE001
            if i == retries:
                return b'', 0, '%r' % (e,)
            time.sleep(2 + 2 * i)
    return b'', 0, 'unreachable'


def norm(s):
    s = unicodedata.normalize('NFKD', s or '').lower()
    return re.sub(r'[^a-z0-9]', '', s)


def safe_name(title, maxlen=70):
    s = re.sub(r'[\\/:*?"<>|\n\r\t]+', ' ', title).strip()
    s = re.sub(r'\s+', ' ', s)
    return s[:maxlen].rstrip()


def thread_dir(p):
    for d in BM.iterdir():
        if d.is_dir() and d.name.startswith(p.get('thread', '??') + '_'):
            return d
    return None


def on_disk(td, title):
    """该篇是否已有 PDF 或 XML 产物。"""
    if not td:
        return None
    key = norm(title)[:38]
    for f in list(td.glob('*.pdf')) + list(td.glob('*.xml')):
        if key in norm(f.stem) or norm(f.stem)[:38] in key:
            return f
    return None


def validate(body, kind):
    if not body:
        return False, '空响应'
    if kind == 'xml':
        head = body[:200000].decode('utf-8', 'ignore')
        if '<body' not in head:
            m = re.search(r'<title>(.*?)</title>', head, re.S)
            return False, '不含 <body>（疑似错误页）%s' % (m.group(1)[:60] if m else '')
        return True, '%d B JATS' % len(body)
    if not body.startswith(b'%PDF'):
        txt = body[:3000].decode('utf-8', 'ignore')
        for m in BLOCK_MARKERS:
            if m in txt:
                return False, '反爬验证页（%s）' % m
        return False, '非 PDF 响应（%d B，%s）' % (len(body), txt[:70].replace('\n', ' '))
    if len(body) < 20000:
        return False, 'PDF 过小 %d B' % len(body)
    if b'%%EOF' not in body[-4096:]:
        return False, 'PDF 尾部无 %%EOF（截断）'
    return True, '%d B' % len(body)


def try_epmc(p):
    """Europe PMC 按 DOI 或题名查，拿官方 JATS 全文 XML。"""
    doi = (p.get('doi') or '').strip()
    queries = []
    if doi:
        queries.append('DOI:"%s"' % doi)
    t = (p.get('title') or '').strip()
    if len(t) > 20:
        queries.append('TITLE:"%s"' % t[:160])
    for q in queries:
        url = '%s?query=%s&format=json&pageSize=3&resultType=core' % (EPMC, urllib.parse.quote(q))
        body, st, _ = get(url, timeout=40)
        if st != 200 or not body:
            continue
        try:
            res = json.loads(body.decode('utf-8', 'ignore')).get('resultList', {}).get('result', [])
        except Exception:                                        # noqa: BLE001
            continue
        for r in res:
            pmcid = r.get('pmcid')
            if not pmcid:
                continue
            # 题名相似度门槛，避免抓到同 DOI 的姊妹记录
            if doi and (r.get('doi') or '').lower() != doi.lower():
                continue
            # 2026-10-01 实测：fullTextXML 只服务 isOpenAccess=Y 的子集。
            # inEPMC=Y 但 isOA=N（如 PMC 里的 author manuscript）会返回 HTTP 500，
            # 而 NCBI 的 PMC PDF 链接是 JS 门禁页（返回 HTML），故此处一律跳过。
            if r.get('isOpenAccess') != 'Y':
                continue
            xb, xs, _ = get(EPMC_XML % pmcid, timeout=90)
            ok, why = validate(xb, 'xml')
            if ok:
                return xb, 'Europe PMC fullTextXML pmcid=%s' % pmcid
    return None, 'Europe PMC 无可取的 OA 全文记录'


def try_url(url, kind='pdf'):
    if not url:
        return None, '无直链'
    body, st, final = get(url, timeout=120)
    ok, why = validate(body, kind)
    if ok:
        return body, '直链 %s' % (final or url)[:100]
    # 直链返回 HTML 落地页时，找页面里的 PDF 链接
    if kind == 'pdf' and body and b'%PDF' not in body[:8]:
        txt = body.decode('utf-8', 'ignore')
        m = re.findall(r'href="([^"]+\.pdf[^"]*)"', txt, re.I)
        if m:
            cand = urllib.parse.urljoin(final or url, m[0])
            b2, s2, f2 = get(cand, timeout=120)
            ok2, why2 = validate(b2, 'pdf')
            if ok2:
                return b2, '落地页内 PDF %s' % f2[:90]
            return None, '落地页内 PDF 亦失败: %s' % why2[:60]
    return None, why


def probe_epmc(p):
    """只查不下载：问清这篇在 Europe PMC 的可取状态。"""
    doi = (p.get('doi') or '').strip()
    if not doi:
        return {'epmc': 'no_doi'}
    url = '%s?query=%s&format=json&pageSize=2&resultType=core' % (EPMC, urllib.parse.quote('DOI:"%s"' % doi))
    body, st, _ = get(url, timeout=30, retries=1)
    if st != 200 or not body:
        return {'epmc': 'query_failed'}
    try:
        res = json.loads(body.decode('utf-8', 'ignore')).get('resultList', {}).get('result', [])
    except Exception:                                            # noqa: BLE001
        return {'epmc': 'bad_json'}
    for r in res:
        if (r.get('doi') or '').lower() == doi.lower():
            return {'epmc': 'hit', 'pmcid': r.get('pmcid', ''), 'isOA': r.get('isOpenAccess', ''),
                    'inEPMC': r.get('inEPMC', '')}
    return {'epmc': 'no_record'}


def main():
    argv = [a for a in sys.argv[1:] if a != '--probe']
    only_status = argv or None
    probe = '--probe' in sys.argv
    idx = json.loads((BM / 'index.json').read_text(encoding='utf-8'))
    papers = idx['papers']
    results = []
    for n, p in enumerate(papers, 1):
        if only_status and (p.get('oa_status') or '') not in only_status:
            continue
        td = thread_dir(p)
        title = p.get('title') or ''
        if not title.strip():
            results.append({'title': '(空题名)', 'status': 'skip_empty_title',
                            'thread': p.get('thread'), 'note': p.get('note', '')[:80]})
            continue
        exist = on_disk(td, title)
        if exist:
            results.append({'title': title, 'status': 'already', 'path': str(exist.relative_to(ROOT))})
            continue

        if probe:
            pr = probe_epmc(p)
            results.append({'title': title, 'thread': p['thread'], 'status': 'probe',
                            'venue': p.get('venue', ''), 'doi': p.get('doi', ''),
                            'year': p.get('year'), 'oa_status': p.get('oa_status', ''),
                            'is_oa': p.get('is_oa'), 'evidence': p.get('evidence_grade', ''),
                            'oa_landing': p.get('oa_landing', ''), 'url_abs': p.get('url_abs', ''),
                            **pr})
            print('[%3d] %-9s %-56s %s' % (n, pr.get('epmc', ''), title[:56], pr.get('pmcid', '')), flush=True)
            continue

        attempts, got, how = [], None, ''
        # 通道 1：Europe PMC 官方全文
        body, why = try_epmc(p)
        attempts.append('epmc: ' + why[:70])
        if body:
            got, how = body, 'xml'
        # 通道 2/3：已记录的 OA 直链
        if not got:
            for u, k in ((p.get('oa_pdf'), 'pdf'), (p.get('url_pdf'), 'pdf')):
                if u:
                    b, why = try_url(u, k)
                    attempts.append('oa_pdf: ' + why[:70])
                    if b:
                        got, how = b, 'pdf'
                        break
        # 通道 4：arXiv
        if not got and p.get('arxiv_id'):
            b, _, _ = get('https://arxiv.org/pdf/%s' % p['arxiv_id'], timeout=120)
            ok, why = validate(b, 'pdf')
            attempts.append('arxiv: ' + why[:60])
            if ok:
                got, how = b, 'pdf'

        if got and td:
            ext = '.xml' if how == 'xml' else '.pdf'
            dst = td / (safe_name(title) + ext)
            dst.write_bytes(got)
            results.append({'title': title, 'thread': p['thread'],
                            'venue': p.get('venue', ''), 'doi': p.get('doi', ''),
                            'status': 'ok', 'kind': how,
                            'path': str(dst.relative_to(ROOT)), 'bytes': len(got),
                            'via': [a for a in attempts if ': ' in a and not a.endswith('失败: ')]})
            print('[%3d] ✅ %-4s %-58s %s' % (n, how, title[:58], dst.name[:40]), flush=True)
        else:
            results.append({'title': title, 'thread': p['thread'],
                            'venue': p.get('venue', ''), 'doi': p.get('doi', ''),
                            'year': p.get('year'), 'oa_status': p.get('oa_status', ''),
                            'is_oa': p.get('is_oa'), 'evidence': p.get('evidence_grade', ''),
                            'status': 'blocked_or_paywalled',
                            'oa_landing': p.get('oa_landing', ''),
                            'url_abs': p.get('url_abs', ''), 'attempts': attempts})
            print('[%3d] ⛔ %-58s %s' % (n, title[:58], (attempts[-1] if attempts else '')[:50]), flush=True)

    prev = {}
    if REPORT.exists():
        try:
            prev = {r.get('title'): r for r in json.loads(REPORT.read_text(encoding='utf-8')).get('items', [])}
        except Exception:                                        # noqa: BLE001
            prev = {}
    for r in results:
        if r['status'] in ('ok', 'already'):
            prev[r['title']] = r
    items = list(prev.values())
    REPORT.write_text(json.dumps({'generated': 'bm3_fetch.py（隔离版，不写共享下载报告）',
                                  'items': items}, ensure_ascii=False, indent=1), encoding='utf-8')
    from collections import Counter
    print('\n汇总：%s' % dict(Counter(r['status'] for r in items)))
    print('报告：%s' % REPORT.relative_to(ROOT))


if __name__ == '__main__':
    main()
