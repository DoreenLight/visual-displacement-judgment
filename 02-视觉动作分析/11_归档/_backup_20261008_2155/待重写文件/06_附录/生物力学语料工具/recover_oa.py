#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recover_oa.py — 语料三缺全文条目的**替代 OA 通路**恢复

为什么需要：
    `download_oa.py` 之后仍有 66 篇缺全文：
      · 25 篇 Unpaywall 确认有官方 OA 直链，但出版商反爬拦截（MDPI 的 bm-verify 令牌页、
        Springer/BMC 的 JS 挑战页、Wiley 403）；
      · 41 篇 Unpaywall 判为非 OA。
    但**「出版商站点取不到」≠「没有合法开放版本」**。还有两条合规通路没用：
      1. **Europe PMC** —— 生物医学类论文的官方 OA 镜像（按 DOI 检索，给出全文链接）
      2. **arXiv 预印本** —— 工程/计算机类论文常先发预印本；**同一工作**，
         预印本可合法作为「待与正式版核对」的替代来源

纪律（与语料二工具链一致）：
  · 只走官方 OA 位置与作者公开预印本；**不绕付费墙、不破反爬、不用影子图书馆**
  · 命中 arXiv 预印本的条目**必须打 `preprint_of` 标记**，提示「与正式版核对后方可引用数值」
  · 每条记录来源类型与命中通路，写进下载报告，**口径可复算**

用法：python3 recover_oa.py [--limit N] [--sleep 1.0]
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = (HERE / '../../01_参考论文/生物力学线').resolve()
INDEX = DEST / 'index.json'
DL_REPORT = HERE / 'biomech_download_report.json'
REPORT = HERE / 'biomech_recovery_report.json'
UA = 'kb-research/1.0 (mailto:kb-research@example.org)'

EPMC = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
ARXIV = 'https://export.arxiv.org/api/query'


def get(url, timeout=45, retries=3, headers=None):
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(45, 3 * (2 ** a)))
                continue
            time.sleep(1.2 * (a + 1))
        except Exception as e:                                    # noqa: BLE001
            last = e
            time.sleep(1.2 * (a + 1))
    raise RuntimeError('请求失败 %s -> %r' % (url[:90], last))


def norm(s):
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).split())


def similar(a, b):
    import difflib
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def via_europepmc(doi):
    """按 DOI 查 Europe PMC。返回 (is_oa, pdf_url, note)。"""
    if not doi:
        return None, '', '无 DOI'
    q = 'DOI:"%s"' % doi
    url = '%s?query=%s&format=json&pageSize=3&resultType=core' % (EPMC, urllib.parse.quote(q))
    try:
        d = json.loads(get(url).decode('utf-8', 'replace'))
    except Exception:                                           # noqa: BLE001
        return None, '', 'EuropePMC 查询失败'
    res = (d.get('resultList') or {}).get('result') or []
    if not res:
        return None, '', 'EuropePMC 无记录'
    r = res[0]
    is_oa = r.get('isOpenAccess') == 'Y'
    pmcid = r.get('pmcid') or ''
    src = r.get('source') or ''
    # 实测：`?pdf=render` 403、REST `fullTextPdf` 404，但 **`fullTextXML` 正常返回完整 JATS 正文**。
    # 对本项目而言 XML 比 PDF 更有用：可取到结构化正文、表格与具体数值定义（如坐标轴、ROM 限值）。
    xml = ''
    if pmcid and is_oa:
        xml = 'https://www.ebi.ac.uk/europepmc/webservices/rest/%s/fullTextXML' % pmcid
    return is_oa, xml, 'EuropePMC pmcid=%s src=%s isOA=%s' % (pmcid or '—', src or '—', r.get('isOpenAccess'))


def via_arxiv(title):
    """按标题查 arXiv 预印本。冒号会让 ti: 检索落空，故同时用去冒号的尾部。"""
    import difflib
    cands = []
    for query in ('ti:"%s"' % title, 'all:"%s"' % re.sub(r'[:：]', ' ', title)):
        url = '%s?search_query=%s&max_results=6' % (ARXIV, urllib.parse.quote(query))
        try:
            x = get(url).decode('utf-8', 'replace')
        except Exception:                                       # noqa: BLE001
            continue
        for blk in re.findall(r'<entry>(.*?)</entry>', x, re.S):
            tm = re.search(r'<title>(.*?)</title>', blk, re.S)
            idm = re.search(r'<id>http://arxiv.org/abs/([^<]+)</id>', blk)
            if not tm or not idm:
                continue
            ti = ' '.join(re.sub(r'<[^>]+>', '', tm.group(1)).split())
            cands.append((idm.group(1), ti))
        time.sleep(3.4)
    best = None
    for aid, ti in cands:
        r = difflib.SequenceMatcher(None, norm(title), norm(ti)).ratio()
        if r >= 0.80 and (best is None or r > best[0]):
            best = (r, aid, ti)
    if not best:
        return '', '', 'arXiv 无相似预印本'
    return best[1], 'https://arxiv.org/pdf/%s' % best[1], \
        'arXiv 预印本 %s，标题相似度 %.3f —— 需与正式版核对后方可引用数值' % (best[1], best[0])


def validate(path, kind):
    """PDF 与 JATS XML 两种工件的校验不能混用。"""
    try:
        size = path.stat().st_size
    except OSError:
        return False, 'stat 失败', 0
    if kind == 'europepmc_fulltext_xml':
        if size < 8 * 1024:
            return False, 'xml_too_small:%d' % size, size
        head = path.read_bytes()[:4000]
        if b'<?xml' not in head[:200]:
            return False, 'not_xml', size
        try:
            txt = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:                                       # noqa: BLE001
            return False, 'read 失败', size
        if '<body' not in txt:
            return False, 'xml_no_body（可能是错误页或仅元数据）', size
        return True, 'ok', size
    if size < 40 * 1024:
        return False, 'too_small:%d' % size, size
    with open(path, 'rb') as f:
        if f.read(4) != b'%PDF':
            return False, 'not_pdf_magic', size
    import subprocess
    try:
        out = subprocess.run(['file', '-b', str(path)], capture_output=True, text=True, timeout=25).stdout
    except Exception:                                           # noqa: BLE001
        return False, 'file(1) 不可用', size
    return (True, 'ok', size) if 'PDF document' in out else (False, 'file1_not_pdf', size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--sleep', type=float, default=1.0)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    idx = json.loads(INDEX.read_text(encoding='utf-8'))
    tn = idx['meta']['axes']['thread']
    have = set()
    for p in idx['papers']:
        have.add((p['thread'], p.get('doi') or p.get('arxiv_id') or p['title'][:60]))

    dl = {}
    if DL_REPORT.exists():
        for it in json.loads(DL_REPORT.read_text(encoding='utf-8'))['items']:
            dl[it['key']] = it

    import build_biomech as B
    todo = []
    for p in idx['papers']:
        key = p.get('doi') or p.get('arxiv_id') or p['title'][:60]
        disk = DEST / ('%s_%s' % (p['thread'], tn[p['thread']])) / \
            ('%s.pdf' % B.safe_name(B.title_of(p), B.NAME_MAXLEN))
        xml = disk.with_suffix('.xml')
        if disk.exists() and disk.stat().st_size > 0:
            continue
        if xml.exists() and xml.stat().st_size > 0:
            continue                       # 已有 Europe PMC 全文 XML
        todo.append(p)
    if args.limit:
        todo = todo[:args.limit]
    print('待恢复：%d 篇（已入库 %d 篇）' % (len(todo), len(idx['papers']) - len(todo)))

    items, found = [], 0
    for i, p in enumerate(todo):
        rec = {'title': p['title'], 'thread': p['thread'],
               'doi': p.get('doi', ''), 'arxiv_id': p.get('arxiv_id', ''),
               'routes': [], 'status': 'not_recovered', 'note': ''}
        # 通路 1：Europe PMC（按 DOI）
        try:
            is_oa, pdf, note = via_europepmc(p.get('doi', ''))
            rec['routes'].append({'via': 'europepmc', 'is_oa': is_oa, 'note': note})
            if is_oa and pdf:
                rec['pmc_xml'] = pdf
        except Exception as e:                                   # noqa: BLE001
            rec['routes'].append({'via': 'europepmc', 'note': 'ERR %r' % (e,)})
        time.sleep(args.sleep)
        # 通路 2：arXiv 预印本
        try:
            aid, apdf, anote = via_arxiv(p['title'])
            rec['routes'].append({'via': 'arxiv', 'note': anote})
            if aid:
                rec['preprint_arxiv_id'] = aid
                rec['preprint_of'] = '正式版 DOI %s' % (p.get('doi') or p.get('arxiv_id') or '—')
                rec['preprint_note'] = anote
                rec['preprint_pdf'] = apdf
        except Exception as e:                                   # noqa: BLE001
            rec['routes'].append({'via': 'arxiv', 'note': 'ERR %r' % (e,)})

        # 取件：优先 PMC 正式全文，其次预印本
        url, kind = (rec.get('pmc_xml'), 'europepmc_fulltext_xml') if rec.get('pmc_xml') \
            else (rec.get('preprint_pdf'), 'arxiv_preprint')
        if url and not args.dry_run:
            sub = DEST / ('%s_%s' % (p['thread'], tn[p['thread']]))
            ext = '.xml' if kind == 'europepmc_fulltext_xml' else '.pdf'
            out = sub / ('%s%s' % (B.safe_name(B.title_of(p), B.NAME_MAXLEN), ext))
            tmp = out.with_suffix(ext + '.part')
            try:
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                                  '(KHTML, like Gecko) Chrome/124.0 Safari/537.36',
                    'Accept': 'application/pdf,text/html,*/*'})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                tmp.write_bytes(data)
                ok, reason, size = validate(tmp, kind)
                if ok:
                    tmp.rename(out)
                    rec['status'] = 'recovered'
                    rec['recovered_via'] = kind
                    rec['path'] = str(out.relative_to(DEST))
                    rec['bytes'] = size
                    found += 1
                else:
                    rec['status'] = 'fetch_failed'
                    rec['note'] = reason
                    if tmp.exists():
                        tmp.unlink()
            except Exception as e:                               # noqa: BLE001
                rec['status'] = 'fetch_failed'
                rec['note'] = repr(e)[:120]
            time.sleep(1.2)
        elif url and args.dry_run:
            rec['status'] = 'would_recover'
            rec['recovered_via'] = kind

        items.append(rec)
        print('[%3d/%3d] %-14s %-56s %s' % (
            i + 1, len(todo), rec['status'], p['title'][:56],
            (rec.get('recovered_via') or rec.get('note') or '')[:34]), flush=True)
        REPORT.write_text(json.dumps({'policy': '仅官方 OA 与作者公开预印本；不绕付费墙、不破反爬',
                                      'items': items}, ensure_ascii=False, indent=2), encoding='utf-8')

    from collections import Counter
    print('\n恢复 %d / %d 篇' % (found, len(todo)))
    print('状态：%s' % dict(Counter(i['status'] for i in items)))
    print('通路：%s' % dict(Counter(i.get('recovered_via', '—') for i in items)))
    pp = sum(1 for i in items if i.get('preprint_arxiv_id'))
    print('其中发现 arXiv 预印本：%d 篇（**须与正式版核对后方可引用数值**）' % pp)


if __name__ == '__main__':
    main()
