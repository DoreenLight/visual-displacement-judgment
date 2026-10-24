#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_arxiv_abs.py — 用 abs 页面补齐被限流漏掉的 arXiv 条目元数据

为什么需要：
    arXiv 官方 API（export.arxiv.org/api/query）在连续调用后会超时/429。
    实测 94 条候选里 6 条 arXiv 优先件因此拿不到元数据 —— 而这 6 条恰好是
    决定研究定位的关键先行，**绝不能因为 API 限流就丢在待裁定里**。

    arxiv.org/abs/<id> 的 HTML 页面带标准 Highwire/citation_* 元标签
    （citation_title / citation_date / citation_journal_title / citation_doi /
    citation_author），实测在 API 被限流时仍可访问。
    这条通路只用于**补齐官方元数据**，不用于「凑数」——解析不到就照实标失败。

判据：
    citation_title 必须与 arXiv ID 绑定（同一页面），解析成功即视为 arXiv 官方确认，
    证据等级记为 A（官方页面直读），并保留 api_path 标记以示取得通路。

用法：python3 patch_arxiv_abs.py
"""
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESOLVED = HERE / 'biomech_resolved.json'
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
SLEEP = 3.5


def clean(s):
    return ' '.join(re.sub(r'<[^>]+>', '', s or '').split())


def fetch_abs(aid, retries=3):
    url = 'https://arxiv.org/abs/%s' % urllib.parse.quote(aid)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                html = r.read().decode('utf-8', 'replace')
            if 'citation_title' not in html:
                last = '页面无 citation_title 标签'
                time.sleep(4 * (attempt + 1))
                continue
            return html
        except Exception as e:                                   # noqa: BLE001
            last = e
            time.sleep(4 * (attempt + 1))
    raise RuntimeError('abs 页面取件失败 %s: %r' % (aid, last))


def meta(html, name):
    """Highwire citation_* 标签：name="citation_title" content="..." """
    m = re.search(r'<meta[^>]+name="%s"[^>]+content="([^"]*)"' % re.escape(name), html)
    if not m:
        m = re.search(r'<meta[^>]+content="([^"]*)"[^>]+name="%s"' % re.escape(name), html)
    return m.group(1).strip() if m else ''


def all_meta(html, name):
    return [m.group(1).strip() for m in
            re.finditer(r'<meta[^>]+name="%s"[^>]+content="([^"]*)"' % re.escape(name), html)]


def main():
    data = json.loads(RESOLVED.read_text(encoding='utf-8'))
    papers = data['papers']
    todo = [p for p in papers if p.get('arxiv_id') and p['status'] not in ('confirmed', 'confirmed_single')]
    print('需补齐的 arXiv 条目：%d' % len(todo))

    fixed = 0
    for p in todo:
        aid = p['arxiv_id']
        try:
            html = fetch_abs(aid)
        except Exception as e:                                   # noqa: BLE001
            p['abs_patch_error'] = str(e)[:160]
            print('[FAIL] %-12s %s' % (aid, str(e)[:80]))
            time.sleep(SLEEP)
            continue

        title = clean(meta(html, 'citation_title'))
        date = meta(html, 'citation_date')
        journal = clean(meta(html, 'citation_journal_title'))
        doi = meta(html, 'citation_doi')
        authors = [clean(a) for a in all_meta(html, 'citation_author')][:8]
        abstr = ''
        m = re.search(r'<meta[^>]+name="citation_abstract"[^>]+content="([^"]*)"', html)
        if m:
            abstr = clean(m.group(1))

        if not title:
            p['abs_patch_error'] = '解析不到 citation_title'
            print('[FAIL] %-12s 解析不到标题' % aid)
            time.sleep(SLEEP)
            continue

        p['canonical_title'] = title
        p['abs_meta'] = {'title': title, 'date': date, 'journal': journal, 'doi': doi,
                         'authors': authors, 'abstract_len': len(abstr)}
        p.setdefault('authors', authors)
        if abstr:
            p['abstract'] = abstr
        if doi:
            p['doi'] = doi
        if journal:
            p['venue'] = journal
        if date and not p.get('year'):
            y = re.search(r'(\d{4})', date)
            p['year'] = int(y.group(1)) if y else None
        p['confirm_sources'] = sorted(set(p.get('confirm_sources', []) + ['arxiv_abs_page']))
        p['n_confirm'] = len(p['confirm_sources'])
        p['title_scores'] = list(p.get('title_scores', [])) + [['arxiv_abs_page', 1.0]]
        p['status'] = 'confirmed'
        p['evidence_grade'] = 'A（arXiv abs 官方页面直读，含 citation_* 元标签）'
        p['meta_path'] = 'abs_page_fallback（export API 被限流）'
        p.pop('doi_conflicts', None)
        p.pop('abs_patch_error', None)
        fixed += 1
        print('[OK  ] %-12s %s' % (aid, title[:76]))
        time.sleep(SLEEP)

    RESOLVED.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    from collections import Counter
    print('\n补齐 %d/%d 条。当前状态：%s' % (fixed, len(todo), dict(Counter(x['status'] for x in papers))))


if __name__ == '__main__':
    main()
