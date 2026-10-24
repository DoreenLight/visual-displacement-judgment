#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resolve_corpus.py — 用 arXiv 官方 API 把「预期标题」解析为规范元数据。

设计原则（对应交付规范「证据可核验」）：
  1. 不硬编码 arXiv ID（避免 ID 臆造）；以 ti: 精确标题检索取得规范记录。
  2. 逐条做标题相似度比对，低于阈值不自动采信，写入 review / failed 供人工复核。
  3. 输出的每条记录都带 arxiv_id、标题、作者、日期、分类、摘要、期刊参考与相似度分。

用法：
    python3 resolve_corpus.py                 # 增量（已有结果则跳过已确认条目）
    python3 resolve_corpus.py --force         # 全部重查
    python3 resolve_corpus.py --only 3,7,12   # 只查指定序号
"""
import argparse
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAND = HERE / 'corpus_candidates.json'
OUT = HERE / 'corpus_resolved.json'

UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
API = 'https://export.arxiv.org/api/query'
SLEEP = 3.2          # arXiv 官方建议 ≥3 秒/请求
AUTO_OK = 0.90       # ≥此值自动采信
NEED_REVIEW = 0.74  # 介于两者之间 → 人工复核


def norm(s):
    s = (s or '').lower()
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return ' '.join(s.split())


def query_variants(title):
    """arXiv 的 ti:"..." 短语检索遇到冒号会整条落空（实测 PoseFormer/METRO/Performer 均返回空），
    故按置信度从高到低给出多个检索变体。"""
    t = title.strip()
    variants = [('ti-full', 'ti:"%s"' % t)]
    if ':' in t:
        head, tail = t.split(':', 1)
        head, tail = head.strip(), tail.strip()
        if len(tail.split()) >= 3:
            variants.append(('ti-tail', 'ti:"%s"' % tail))
        if len(head.split()) >= 2:
            variants.append(('ti-head', 'ti:"%s"' % head))
    words = [w for w in re.sub(r'[^A-Za-z0-9 ]+', ' ', t).split() if len(w) > 2]
    if len(words) >= 4:
        variants.append(('ti-words', 'ti:"%s"' % ' '.join(words[:6])))
        variants.append(('all-words', 'all:"%s"' % ' '.join(words[:5])))
    return variants


class NoResults(RuntimeError):
    """所有检索变体都成功返回、但均无匹配条目 —— 属「arXiv 未收录」，不是网络故障。"""


def fetch_titles(title, retries=3):
    """按标题检索 arXiv，返回 (查询方式, xml 文本)。命中任一变体即返回。
    全部变体无结果 → NoResults；有网络故障且未命中 → RuntimeError。"""
    last = None
    saw_network_error = False
    for tag, q in query_variants(title):
        url = '%s?search_query=%s&max_results=8' % (API, urllib.parse.quote(q))
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': UA})
                with urllib.request.urlopen(req, timeout=40) as r:
                    xml = r.read().decode('utf-8', 'replace')
                if '<entry>' in xml:
                    return tag, xml
                last = '%s 无结果' % tag
                break                      # 无结果就不必重试该变体
            except Exception as e:         # noqa: BLE001
                saw_network_error = True
                last = '%s: %r' % (tag, e)
                time.sleep(3 * (attempt + 1))
        time.sleep(SLEEP)
    if saw_network_error:
        raise RuntimeError('arXiv 检索故障: %s' % last)
    raise NoResults('arXiv 无此条目（所有检索变体均无结果）: %s' % last)


def parse_entries(xml):
    """解析 Atom feed，返回条目列表。"""
    out = []
    for block in re.findall(r'<entry>(.*?)</entry>', xml, re.S):
        def grab(tag, src):
            m = re.search(r'<%s[^>]*>(.*?)</%s>' % (tag, tag), src, re.S)
            if not m:
                return ''
            v = re.sub(r'<[^>]+>', '', m.group(1))
            v = (v.replace('&amp;', '&').replace('&lt;', '<')
                  .replace('&gt;', '>').replace('&quot;', '"').replace('&#39;', "'"))
            return ' '.join(v.split())

        aid = grab('id', block)
        aid = aid.rsplit('/abs/', 1)[-1] if '/abs/' in aid else aid
        cats = re.findall(r'<category term="([^"]+)"', block)
        out.append({
            'arxiv_id': aid,
            'title': grab('title', block),
            'authors': re.findall(r'<name>(.*?)</name>', block),
            'published': grab('published', block),
            'updated': grab('updated', block),
            'summary': grab('summary', block),
            'primary_category': (cats[0] if cats else ''),
            'categories': cats,
            'journal_ref': grab('arxiv:journal_ref', block),
            'doi': (re.search(r'<arxiv:doi[^>]*>(.*?)</arxiv:doi>', block, re.S).group(1).strip()
                    if re.search(r'<arxiv:doi[^>]*>(.*?)</arxiv:doi>', block, re.S) else ''),
            'comment': grab('arxiv:comment', block),
        })
    return out


def best_match(want, entries):
    wn = norm(want)
    scored = []
    for e in entries:
        r = difflib.SequenceMatcher(None, wn, norm(e['title'])).ratio()
        scored.append((r, e))
    scored.sort(key=lambda x: -x[0])
    return scored[0] if scored else (0.0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--only', default='')
    args = ap.parse_args()

    data = json.loads(CAND.read_text(encoding='utf-8'))
    papers = data['papers']
    only = {int(x) for x in args.only.split(',') if x.strip().isdigit()}

    prev = {}
    if OUT.exists() and not args.force:
        try:
            prev = {p['query_title']: p for p in json.loads(OUT.read_text(encoding='utf-8'))['papers']}
        except Exception:                            # noqa: BLE001
            prev = {}

    results = []
    for i, p in enumerate(papers):
        # 缓存键用「编制时的标题」而非序号：这样新增条目或修正标题都不会错位复用旧结果。
        # --only 表示「强制重查这些序号」，其余仍走缓存；结果列表始终与 papers 等长等序
        key = p['title']
        if not args.force and i not in only and key in prev and prev[key].get('status') in ('confirmed', 'rejected'):
            prev[key]['idx'] = i
            results.append(prev[key]); continue

        try:
            qtag, xml = fetch_titles(p['title'])
            entries = parse_entries(xml)
            score, ent = best_match(p['title'], entries)
        except NoResults as e:                   # noqa: PERF203
            results.append({**p, 'idx': i, 'status': 'unresolved', 'match_score': 0.0,
                            'reason': str(e), 'candidates': []})
            print('[%3d] NOARXIV  %s' % (i, p['title'][:56]), flush=True)
        except Exception as e:                      # noqa: BLE001
            results.append({**p, 'idx': i, 'status': 'error', 'error': str(e), 'match_score': 0.0})
            print('[%3d] ERROR  %s -> %s' % (i, p['title'][:56], e), flush=True)
            time.sleep(SLEEP); continue

        if ent is None or score < NEED_REVIEW:
            results.append({**p, 'idx': i, 'status': 'unresolved', 'match_score': round(score, 3),
                            'candidates': [{'arxiv_id': e['arxiv_id'], 'title': e['title']} for e in entries[:4]]})
            print('[%3d] %-6s %.2f  %s' % (i, 'UNRES', score, p['title'][:56]), flush=True)
        elif score >= AUTO_OK:
            y = int(ent['published'][:4]) if ent['published'] else p.get('year')
            rec = {**p, 'idx': i, 'status': 'confirmed', 'match_score': round(score, 3),
                   'arxiv_id': ent['arxiv_id'], 'canonical_title': ent['title'], 'query': qtag,
                   'authors': ent['authors'], 'published': ent['published'],
                   'updated': ent['updated'], 'primary_category': ent['primary_category'],
                   'categories': ent['categories'], 'journal_ref': ent['journal_ref'],
                   'doi': ent['doi'], 'comment': ent['comment'], 'abstract': ent['summary'],
                   'api_year': y, 'year_mismatch': (y != p.get('year'))}
            results.append(rec)
            flag = '  ⚠年份' if rec['year_mismatch'] else ''
            print('[%3d] OK     %.2f  %-52s %s%s' % (i, score, ent['title'][:52], ent['arxiv_id'], flag), flush=True)
        else:
            results.append({**p, 'idx': i, 'status': 'review', 'match_score': round(score, 3),
                            'arxiv_id': ent['arxiv_id'], 'canonical_title': ent['title'],
                            'published': ent['published'], 'primary_category': ent['primary_category'],
                            'journal_ref': ent['journal_ref'], 'abstract': ent['summary'],
                            'candidates': [{'arxiv_id': e['arxiv_id'], 'title': e['title']} for e in entries[:4]]})
            print('[%3d] %-6s %.2f  %-52s %s' % (i, 'REVIEW', score, ent['title'][:52], ent['arxiv_id']), flush=True)

        OUT.write_text(json.dumps({'meta': data['meta'], 'papers': results}, ensure_ascii=False, indent=2), encoding='utf-8')
        time.sleep(SLEEP)

    OUT.write_text(json.dumps({'meta': data['meta'], 'papers': results}, ensure_ascii=False, indent=2), encoding='utf-8')
    from collections import Counter
    c = Counter(r.get('status') for r in results)
    print('\n汇总：%s' % dict(c), file=sys.stderr)


if __name__ == '__main__':
    main()
