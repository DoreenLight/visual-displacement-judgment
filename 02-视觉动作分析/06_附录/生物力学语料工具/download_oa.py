#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_oa.py — 下载语料三的官方开放获取 PDF（三重校验）

只从 Unpaywall 提供的**官方 OA 位置**取件（publisher / PMC / 机构库），
不做任何付费墙绕过、不使用影子图书馆。非 OA 条目一律跳过，
并进入「需人工取件清单」由研究者经学校数据库自行取件
（2026-10-03 起统一收进 `06_附录/待下载论文总清单.md` §2，由 `gen_acquisition_list.py` 生成）。

三重校验（与语料二同规格，防止把 HTML 错误页当 PDF 存入库）：
  1. 文件头必须是 %PDF
  2. 体积下限（默认 40 KB）
  3. file(1) 判定为 PDF document

用法：
    python3 download_oa.py
    python3 download_oa.py --retry-failed
"""
import argparse
import json
import subprocess
import time
import urllib.request
import urllib.error
import urllib.parse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESOLVED = HERE / 'biomech_resolved.json'
REPORT = HERE / 'biomech_download_report.json'
DEST = (HERE / '../../01_参考论文/整合系统/生物力学线').resolve()

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36')
MIN_BYTES = 40 * 1024
SLEEP = 1.5


def browser_headers(referer=''):
    """完整浏览器请求头。

    实测：只带 User-Agent 时，Wiley(onlinelibrary) 与 MDPI 一律 403，
    Springer/BMC 的 /track/pdf 会返回 3KB 空壳页。补齐 Accept / Accept-Language /
    Sec-Fetch-* / Referer 后即可正常取到 —— 这些刊物本就是金色 OA，
    补请求头不是绕过付费墙，只是把请求伪装成正常浏览器访问开放 PDF。
    """
    h = {
        'User-Agent': UA,
        'Accept': 'application/pdf,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
    }
    if referer:
        h['Referer'] = referer
    return h


def alt_urls(doi, url):
    """同一 OA 文章的备用直链模式。

    仅在 Unpaywall 直链失败时启用，且**只在已确认 is_oa 的条目上**使用；
    不涉及任何付费墙来源。
    """
    enc = urllib.parse.quote(doi, safe='')
    out = []
    if 'bmcmusculoskeletdisord' in url or 'biomedcentral' in url:
        out.append(url.replace('/track/pdf/', '/counter/pdf/'))
    if 'link.springer.com' in url or 'springer.com' in url:
        out.append('https://link.springer.com/content/pdf/%s.pdf' % enc)
    if 'frontiersin' in url:
        out.append(re.sub(r'/articles/[^/]+/(pdf|full)', '/articles/%s/pdf' % doi, url))
    if 'mdpi.com' in url:
        # MDPI 的 pdf 直链带 version 参数时反而更易 403，去掉参数再试
        out.append(url.split('?')[0])
    if 'onlinelibrary.wiley.com' in url or 'doi/pdfdirect' in url:
        out.append(url.replace('/doi/pdfdirect/', '/doi/epdf/'))
    if 'plos.org' in url:
        out.append(url)
    seen, uniq = set(), []
    for u in out:
        if u and u not in seen and u != url:
            seen.add(u)
            uniq.append(u)
    return uniq


def validate(path):
    try:
        size = path.stat().st_size
    except OSError as e:
        return False, 'stat 失败 %r' % (e,), 0
    if size < MIN_BYTES:
        return False, 'too_small:%d' % size, size
    with open(path, 'rb') as f:
        if f.read(4) != b'%PDF':
            return False, 'not_pdf_magic', size
    try:
        out = subprocess.run(['file', '-b', str(path)], capture_output=True, text=True, timeout=25).stdout
    except Exception as e:                                       # noqa: BLE001
        return False, 'file(1) 不可用 %r' % (e,), size
    if 'PDF document' not in out:
        return False, 'file1_not_pdf', size
    return True, 'ok', size


def safe_name(s, maxlen=60):   # 必须与 build_biomech.py 的 60 严格一致，否则 PDF↔定位卡配不上
    import re
    s = re.sub(r'[\\/:*?"<>|]', '_', (s or '').strip())
    s = re.sub(r'\s+', ' ', s)
    return s[:maxlen].rstrip(' .')


def fetch(url, out_path, referer='', retries=2):
    """取件 → 校验 → 落盘。返回 (ok, bytes, reason, used_url)。

    失败时先试备用直链模式（仅限已确认 OA 的条目），再放弃。
    """
    tmp = out_path.with_suffix('.pdf.part')
    candidates = [(url, referer)] + [(u, '') for u in alt_urls(doi_of(url), url)]
    last = '未知失败'
    for u, ref in candidates:
        for attempt in range(retries):
            try:
                req = urllib.request.Request(u, headers=browser_headers(ref or ('https://doi.org/' + doi_of(u))))
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                tmp.write_bytes(data)
                ok, reason, size = validate(tmp)
                if ok:
                    tmp.rename(out_path)
                    return True, size, '', u
                last = reason
                if reason.startswith('too_small') or reason == 'not_pdf_magic':
                    break            # 换 URL 模式比重试更有用
            except urllib.error.HTTPError as e:
                last = 'http_%d' % e.code
            except Exception as e:                               # noqa: BLE001
                last = repr(e)[:100]
            time.sleep(2 * (attempt + 1))
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return False, 0, last, ''


def doi_of(url):
    m = re.search(r'10\.\d{4,9}/[^\s?#]+', url or '')
    return m.group(0) if m else ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--retry-failed', action='store_true')
    args = ap.parse_args()

    data = json.loads(RESOLVED.read_text(encoding='utf-8'))
    idx_path = DEST / 'index.json'
    if not idx_path.exists():
        print('请先运行 build_biomech.py 生成 index.json', file=__import__('sys').stderr)
        return
    idx = json.loads(idx_path.read_text(encoding='utf-8'))
    thread_names = idx['meta']['axes']['thread']

    prev = {}
    if REPORT.exists() and not args.retry_failed:
        try:
            prev = {i['key']: i for i in json.loads(REPORT.read_text(encoding='utf-8'))['items']}
        except Exception:                                       # noqa: BLE001
            prev = {}

    items = []
    for p in idx['papers']:
        key = p.get('doi') or p.get('arxiv_id') or p['title'][:60]
        urls = []
        if p.get('oa_pdf'):
            urls.append(p['oa_pdf'])
        if p.get('arxiv_id'):
            # arXiv 官方 PDF 端点：比 Unpaywall 链更稳，且对预印本一定是原文
            ax = 'https://arxiv.org/pdf/%s' % p['arxiv_id']
            if ax not in urls:
                urls.append(ax)
        url = urls[0] if urls else ''
        sub = DEST / ('%s_%s' % (p['thread'], thread_names[p['thread']]))
        sub.mkdir(parents=True, exist_ok=True)
        out = sub / ('%s.pdf' % safe_name(p['title']))
        # 磁盘上已有合法 PDF 就直接认账：--retry-failed 重测曾把已成功条目误标为失败
        if out.exists():
            ok_disk, rsn, sz = validate(out)
            if ok_disk:
                items.append({'key': key, 'title': p['title'], 'thread': p['thread'],
                              'status': 'ok', 'path': str(out.relative_to(DEST)), 'bytes': sz,
                              'url': prev.get(key, {}).get('url', ''), 'used_url': '',
                              'reason': '磁盘已存在且通过校验'})
                continue
        if not url:
            items.append({'key': key, 'title': p['title'], 'status': 'not_oa',
                          'reason': 'Unpaywall 未提供 OA PDF 直链', 'path': '', 'bytes': 0, 'url': ''})
            continue
        ok, size, reason, used = False, 0, '无 OA 直链', ''
        for u in urls:
            ok, size, reason, used = fetch(u, out, referer=p.get('oa_landing', ''))
            if ok:
                break
        items.append({'key': key, 'title': p['title'], 'thread': p['thread'],
                      'status': 'ok' if ok else 'failed', 'path': str(out.relative_to(DEST)),
                      'bytes': size, 'url': url, 'tried': urls, 'used_url': used, 'reason': reason})
        print('%-8s %-58s %s' % ('OK' if ok else items[-1]['status'].upper(),
                                  p['title'][:58], ('%d B' % size) if ok else reason[:50]), flush=True)
        time.sleep(SLEEP)
        REPORT.write_text(json.dumps({'items': items}, ensure_ascii=False, indent=2), encoding='utf-8')

    from collections import Counter
    REPORT.write_text(json.dumps(
        {'dest': str(DEST),
         'summary': dict(Counter(i['status'] for i in items)),
         'policy': '仅官方 OA 位置；非 OA 不下载、不绕过付费墙',
         'items': items}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n汇总：%s' % dict(Counter(i['status'] for i in items)))


if __name__ == '__main__':
    main()
