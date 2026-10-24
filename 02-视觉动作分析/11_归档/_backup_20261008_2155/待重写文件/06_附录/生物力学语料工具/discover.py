#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
discover.py — 多源学术发现器（语料三：生物力学线）

为什么需要它：
    现有 resolve_corpus.py 只查 arXiv。但肌肉骨骼动力学 / 上肢康复 / 临床生物力学的
    关键论文绝大多数发表在 J Biomech、Clinical Biomech、IEEE TNSRE、JBHI、
    J Electromyogr Kinesiol、Muscle&Nerve 等期刊，**不在 arXiv**。
    实测：现有 85 篇 Transformer 语料中，muscle/tendon/OpenSim 关键词命中 0 篇、
    inverse dynamics / joint torque 命中 0 篇 —— 这不是「没人做」，是「arXiv 查不到」。

    本脚本做**发现**（seeded multi-source search），把候选池打捞出来交给人工裁定；
    它不自动采信任何一篇。裁定后的清单由 resolve_multisource.py 做**多源交叉证实**。

设计原则（与语料二工具链一致）：
  1. 不硬编码 arXiv ID / DOI；以标题检索取得规范记录。
  2. 每条记录保留「被哪几个独立来源命中」—— 这是后续证据分级的原料。
  3. 只读公开 API，不做付费墙绕过；OA PDF 只从官方 OA 位置取。
  4. 理论搜索空间 / 实际命中 / 实际入库 三者在产物里严格分开。

用法：
    python3 discover.py                        # 全量发现
    python3 discover.py --only B2,B5           # 只跑指定线程
    python3 discover.py --per-query 20         # 每条 query 取多少结果
    python3 discover.py --out biomech_pool.json
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = HERE / 'biomech_seeds.json'

UA = 'kb-research/1.0 (visual-kinematics literature pipeline; contact: local user)'
MAILTO = 'kb-research@example.org'      # OpenAlex polite pool
OPENALEX = 'https://api.openalex.org/works'
CROSSREF = 'https://api.crossref.org/works'
PUBMED_SEARCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
PUBMED_FETCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'

SLEEP_OA = 0.35          # OpenAlex polite pool
SLEEP_CR = 0.5           # Crossref 建议
SLEEP_PM = 0.4           # NCBI 无 key 时 ≤3 req/s


# ------------------------------------------------------------------ 工具函数
def http_json(url, timeout=45, retries=5, sleep=SLEEP_OA):
    """带 429 指数退避的 JSON 请求。

    实测：连续 ~200 次 OpenAlex 请求会触发 HTTP 429。必须显式处理，
    否则 B5 这类后段线程会整段丢失（发现过一次进程在限流中被回收，
    B5 的 18 条 query 只跑完 6 条就没了）。
    """
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode('utf-8', 'replace'))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                wait = min(120, 4 * (2 ** attempt))
                print('      … HTTP %s，退避 %ds 后重试 (%d/%d)'
                      % (e.code, wait, attempt + 1, retries), file=sys.stderr)
                time.sleep(wait)
                continue
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:                                  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError('请求失败 %s -> %r' % (url[:110], last))


def norm(s):
    s = (s or '').lower()
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return ' '.join(s.split())


def inv_index_to_text(inv):
    """OpenAlex 的摘要是倒排索引，需要重建为正常文本。"""
    if not inv:
        return ''
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    if not pos:
        return ''
    out = []
    for i in range(max(pos) + 1):
        out.append(pos.get(i, ''))
    t = ' '.join(out).strip()
    return t[:1400]


# ------------------------------------------------------------------ 各源查询
def q_openalex(query, per_query):
    url = ('%s?search=%s&per-page=%d&mailto=%s'
           % (OPENALEX, urllib.parse.quote(query), per_query, MAILTO))
    d = http_json(url, sleep=SLEEP_OA)
    out = []
    for w in d.get('results', []):
        loc = (w.get('primary_location') or {})
        src = ((loc.get('source') or {}) or {})
        oa = w.get('best_oa_location') or {}
        authors = [a['author']['display_name'] for a in (w.get('authorships') or [])
                   if (a.get('author') or {}).get('display_name')]
        out.append({
            'source': 'openalex',
            'title': w.get('display_name') or w.get('title') or '',
            'year': w.get('publication_year'),
            'doi': (w.get('doi') or '').replace('https://doi.org/', ''),
            'venue': src.get('display_name') or '',
            'authors': authors,
            'cited_by': w.get('cited_by_count', 0),
            'type': w.get('type') or '',
            'abstract': inv_index_to_text(w.get('abstract_inverted_index')),
            'is_oa': (w.get('open_access') or {}).get('is_oa', False),
            'oa_pdf': oa.get('pdf_url') or '',
            'oa_landing': oa.get('landing_page_url') or '',
            'openalex_id': w.get('id', ''),
        })
    time.sleep(SLEEP_OA)
    return out


def q_crossref(query, per_query):
    url = ('%s?query.bibliographic=%s&rows=%d&select=title,DOI,container-title,issued,author,abstract,type,is-referenced-by-count'
           % (CROSSREF, urllib.parse.quote(query), per_query))
    d = http_json(url, sleep=SLEEP_CR)
    out = []
    for w in (d.get('message') or {}).get('items', []):
        t = (w.get('title') or [''])[0]
        if not t:
            continue
        issued = w.get('issued', {}).get('date-parts', [[None]])[0][0]
        out.append({
            'source': 'crossref',
            'title': t,
            'year': issued,
            'doi': w.get('DOI', ''),
            'venue': (w.get('container-title') or [''])[0],
            'authors': [('%s %s' % (a.get('given', ''), a.get('family', ''))).strip()
                        for a in (w.get('author') or [])][:8],
            'cited_by': w.get('is-referenced-by-count', 0),
            'type': w.get('type', ''),
            'abstract': re.sub(r'<[^>]+>', ' ', w.get('abstract') or '').strip()[:1400],
            'is_oa': None, 'oa_pdf': '', 'oa_landing': '', 'openalex_id': '',
        })
    time.sleep(SLEEP_CR)
    return out


def q_pubmed(query, per_query):
    """PubMed 只用于「生物医学线程是否被独立索引」这一交叉验证，不作为主发现源。

    关键：必须加 [Title] 限定。PubMed 默认搜全字段，实测不加限定时
    "muscle redundancy problem motor control" 会返回「uterine mechanics」这类完全无关的条目，
    那样的命中是假信号，会污染后续的交叉证实计数。
    """
    url = ('%s?db=pubmed&term=%s&retmode=json&retmax=%d'
           % (PUBMED_SEARCH, urllib.parse.quote('(%s)[Title]' % query), per_query))
    d = http_json(url, sleep=SLEEP_PM)
    ids = ((d.get('esearchresult') or {}).get('idlist') or [])
    recs = []
    if ids:
        furl = ('%s?db=pubmed&id=%s&retmode=xml' % (PUBMED_FETCH, ','.join(ids)))
        try:
            import urllib.request as _u
            req = _u.Request(furl, headers={'User-Agent': UA})
            with _u.urlopen(req, timeout=45) as r:
                xml = r.read().decode('utf-8', 'replace')
            for blk in re.findall(r'<PubmedArticle>(.*?)</PubmedArticle>', xml, re.S):
                ti = re.search(r'<ArticleTitle>(.*?)</ArticleTitle>', blk, re.S)
                if not ti:
                    continue
                title = re.sub(r'<[^>]+>', '', ti.group(1))
                ab = ' '.join(re.sub(r'<[^>]+>', '', x)
                              for x in re.findall(r'<AbstractText[^>]*>(.*?)</AbstractText>', blk, re.S))
                yr = re.search(r'<Year>(\d{4})</Year>', blk)
                doi = re.search(r'<ArticleId IdType="doi">(.*?)</ArticleId>', blk)
                jr = re.search(r'<Title>(.*?)</Title>', blk, re.S)
                recs.append({
                    'source': 'pubmed',
                    'title': ' '.join(title.split()),
                    'year': int(yr.group(1)) if yr else None,
                    'doi': doi.group(1) if doi else '',
                    'venue': ' '.join(jr.group(1).split()) if jr else '',
                    'authors': [], 'cited_by': 0, 'type': 'journal-article',
                    'abstract': ' '.join(ab.split())[:1400],
                    'is_oa': None, 'oa_pdf': '', 'oa_landing': '',
                    'openalex_id': '', 'pmid_present': True,
                })
        except Exception as e:                                  # noqa: BLE001
            print('    [pubmed efetch 失败] %r' % (e,), file=sys.stderr)
    time.sleep(SLEEP_PM)
    return recs


# ------------------------------------------------------------------ 合并去重
def merge(records):
    """按 DOI 优先、否则按规范化标题合并；记录每个条目被哪些独立来源命中。"""
    buckets = {}

    def key_of(r):
        if r.get('doi'):
            return 'doi:' + r['doi'].lower().strip()
        return 'ti:' + norm(r['title'])[:150]

    for r in records:
        k = key_of(r)
        if k not in buckets:
            buckets[k] = {
                'title': r['title'],
                'year': r.get('year'),
                'doi': r.get('doi', ''),
                'venue': r.get('venue', ''),
                'authors': r.get('authors', []),
                'cited_by': r.get('cited_by', 0),
                'type': r.get('type', ''),
                'abstract': r.get('abstract', ''),
                'sources': [],
                'oa_pdf': '',
                'oa_landing': '',
                'is_oa': False,
                'alt_titles': [],
            }
        b = buckets[k]
        if r['source'] not in b['sources']:
            b['sources'].append(r['source'])
        # 字段补全：空值不覆盖已有值
        for f in ('year', 'doi', 'venue'):
            if not b.get(f) and r.get(f):
                b[f] = r[f]
        if not b['abstract'] and r.get('abstract'):
            b['abstract'] = r['abstract']
        if r.get('cited_by', 0) > b.get('cited_by', 0):
            b['cited_by'] = r['cited_by']
        if r.get('is_oa'):
            b['is_oa'] = True
        if r.get('oa_pdf') and not b['oa_pdf']:
            b['oa_pdf'] = r['oa_pdf']
            b['oa_landing'] = r.get('oa_landing', '')
        # 同一篇被不同源检索到但标题写法不同 -> 留作 alt_titles 供人工裁定
        nt = norm(r['title'])
        if nt != norm(b['title']) and nt not in [norm(x) for x in b['alt_titles']]:
            b['alt_titles'].append(r['title'])
    return list(buckets.values())


def score(b, incl, excl):
    """启发式排序分：标题命中 + 引用量 + 独立来源数 + 近期。"""
    ti = norm(b['title'])
    s = 0.0
    hit = [k for k in incl if k in ti]
    s += 3.0 * len(hit)
    if any(k in ti for k in excl):
        s -= 4.0
    s += min(6.0, (b.get('cited_by') or 0) ** 0.42)          # 引用量，压缩
    s += 1.2 * (len(b['sources']) - 1)                       # 多源交叉命中
    if b.get('is_oa'):
        s += 1.0
    y = b.get('year') or 0
    if y:
        s += 0.8 if y >= 2019 else 0.0
        s += 0.6 if y <= 2005 else 0.0                        # 奠基性工作也不能丢
    if b.get('type') in ('dataset', 'paratext', 'peer-review'):
        s -= 2.0
    return s


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default=str(SEEDS))
    ap.add_argument('--out', default=str(HERE / 'biomech_pool.json'))
    ap.add_argument('--per-query', type=int, default=20)
    ap.add_argument('--only', default='')
    ap.add_argument('--sources', default='openalex,crossref,pubmed',
                    help='逗号分隔：openalex,crossref,pubmed。OpenAlex 连续 ~250 次请求后会严格 429 限流，'
                         '此时可用 --sources crossref,pubmed 续跑，事后回补 OpenAlex。')
    args = ap.parse_args()
    use = {x.strip() for x in args.sources.split(',') if x.strip()}

    seeds = json.loads(Path(args.seeds).read_text(encoding='utf-8'))
    only = {x.strip() for x in args.only.split(',') if x.strip()}
    threads = [t for t in seeds['threads'] if not only or t['id'] in only]

    # --only 重跑时必须与已有池**合并**，否则会把未跑的线程整段覆盖掉
    prev_threads = {}
    outpath = Path(args.out)
    if only and outpath.exists():
        try:
            prev = json.loads(outpath.read_text(encoding='utf-8'))
            prev_threads = {t['id']: t for t in prev.get('threads', [])}
            print('合并模式：已有 %d 个线程（%s），本次只跑 %s'
                  % (len(prev_threads), ','.join(prev_threads), ','.join(sorted(only))), flush=True)
        except Exception:                                       # noqa: BLE001
            prev_threads = {}

    out = {'meta': {**seeds['meta'], 'per_query': args.per_query,
                    'note': '候选池由多源检索生成，未经人工裁定，不得直接当作语料引用'},
           'threads': []}

    for t in threads:
        incl = [k.lower() for k in t.get('include', [])]
        excl = [k.lower() for k in t.get('exclude', [])]
        raw = []
        for q in t['queries']:
            if 'openalex' in use:
                print('  [%s] OA  %s' % (t['id'], q), flush=True)
                try:
                    raw += q_openalex(q, args.per_query)
                except Exception as e:                              # noqa: BLE001
                    print('      ! 失败 %r' % (e,), file=sys.stderr)
            if 'crossref' in use:
                print('  [%s] CR  %s' % (t['id'], q), flush=True)
                try:
                    raw += q_crossref(q, args.per_query)
                except Exception as e:                              # noqa: BLE001
                    print('      ! 失败 %r' % (e,), file=sys.stderr)
        # PubMed 只跑前 3 条 query 做交叉验证信号（控制耗时）
        for q in (t['queries'][:3] if 'pubmed' in use else []):
            print('  [%s] PM  %s' % (t['id'], q), flush=True)
            try:
                raw += q_pubmed(q, 15)
            except Exception as e:                              # noqa: BLE001
                print('      ! 失败 %r' % (e,), file=sys.stderr)

        items = merge(raw)
        for b in items:
            b['score'] = round(score(b, incl, excl), 3)
            b['thread'] = t['id']
            b['thread_name'] = t['name']
        items.sort(key=lambda x: -x['score'])
        by_src = defaultdict(int)
        for b in items:
            for s in b['sources']:
                by_src[s] += 1
        print('\n  == %s %s：原始 %d 条 → 去重 %d 条  %s\n'
              % (t['id'], t['name'], len(raw), len(items), dict(by_src)), flush=True)
        out['threads'].append({
            'id': t['id'], 'name': t['name'], 'purpose': t['purpose'],
            'include': t.get('include', []), 'exclude': t.get('exclude', []),
            'queries': t['queries'], 'raw_hits': len(raw), 'dedup': len(items),
            'items': items,
        })
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    # 回填：本次未跑的线程按原顺序保留
    if prev_threads:
        merged = []
        for t in seeds['threads']:
            cur = next((x for x in out['threads'] if x['id'] == t['id']), None)
            if cur is not None:
                merged.append(cur)
            elif t['id'] in prev_threads:
                merged.append(prev_threads[t['id']])
        out['threads'] = merged

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    tot = sum(t['dedup'] for t in out['threads'])
    print('候选池已写出：%s（%d 线程，去重后共 %d 条）' % (args.out, len(out['threads']), tot))


if __name__ == '__main__':
    main()
