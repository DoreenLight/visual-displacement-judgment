#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resolve_multisource.py — 候选清单的多源交叉证实

与语料二 resolve_corpus.py 的区别：
    语料二只查 arXiv，因为 Transformer 论文几乎都在 arXiv。
    生物力学语料相反：关键词实测 arXiv 命中 0 篇，关键论文在 J Biomech /
    Clinical Biomech / TNSRE / JBHI 等期刊。**只查 arXiv 会把它们全部判成
    「arXiv 未收录」——那不是事实，是工具盲区。**
    故本脚本以 Crossref + OpenAlex 为主，arXiv 为辅，三源互相印证。

判据（沿用语料二的阈值，但作用于多源）：
    标题相似度 ≥0.90 且至少 2 个独立来源确认 → confirmed（证据等级 A）
    标题相似度 ≥0.74 单源确认                  → review（证据等级 B，须人工复核）
    DOI 与来源返回的 DOI 不一致                  → 一律降级为 review 并记录冲突
    全部来源都低于 0.74 或无结果                → unresolved

DOI 冲突单独立为一类 `doi_conflict`：这类是最危险的错误引用
（语料二曾出现登记的 EMBC DOI 实际指向一篇心脏 MRI 论文）。

用法：python3 resolve_multisource.py [--only B1,B5] [--sleep 0.4]
"""
import argparse
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAND = HERE / 'biomech_candidates.json'
OUT = HERE / 'biomech_resolved.json'

UA = 'kb-research/1.0 (visual-kinematics literature verification)'
MAILTO = 'kb-research@example.org'
AUTO_OK = 0.90
NEED_REVIEW = 0.74


def norm(s):
    s = (s or '').lower()
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return ' '.join(s.split())


def http_json(url, timeout=45, retries=4, headers=None):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode('utf-8', 'replace'))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 3 * (2 ** attempt)))
                continue
            time.sleep(1.2 * (attempt + 1))
        except Exception as e:                                  # noqa: BLE001
            last = e
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError('请求失败 %s -> %r' % (url[:100], last))


# ------------------------------------------------------------------ 各源
def from_arxiv(aid):
    xml_url = 'https://export.arxiv.org/api/query?id_list=%s' % urllib.parse.quote(aid)
    for attempt in range(3):
        try:
            req = urllib.request.Request(xml_url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                xml = r.read().decode('utf-8', 'replace')
            break
        except Exception as e:                                  # noqa: BLE001
            if attempt == 2:
                raise RuntimeError('arXiv 失败 %s %r' % (aid, e))
            time.sleep(3 * (attempt + 1))
    m = re.search(r'<entry>(.*?)</entry>', xml, re.S)
    if not m:
        return None
    s = m.group(1)

    def g(tag):
        mm = re.search(r'<%s[^>]*>(.*?)</%s>' % (tag, tag), s, re.S)
        if not mm:
            return ''
        return ' '.join(re.sub(r'<[^>]+>', '', mm.group(1)).split())

    doi = re.search(r'<arxiv:doi[^>]*>(.*?)</arxiv:doi>', s, re.S)
    return {
        'source': 'arxiv', 'title': g('title'),
        'authors': re.findall(r'<name>(.*?)</name>', s),
        'published': g('published'), 'updated': g('updated'),
        'abstract': g('summary'), 'doi': doi.group(1).strip() if doi else '',
        'journal_ref': g('arxiv:journal_ref'),
        'primary_category': (re.findall(r'<category term="([^"]+)"', s) or [''])[0],
        'categories': re.findall(r'<category term="([^"]+)"', s),
    }


def from_crossref_doi(doi):
    """按 DOI 直接取权威记录。

    为什么必须有这条通路：候选清单里**已经带 DOI**，却仍去按标题搜索，
    会发生两件事——
      1. 标题搜索的 top-5 里**可能根本没有正确那条**（实测 ICCVW 2017
         `10.1109/iccvw.2017.104` 官方题名与登记标题相似度 1.000，却被判成 0.885）；
      2. 检索不到就误判成「无来源命中」，把好引用打成待裁定。
    DOI 是权威标识，**有 DOI 就该先按 DOI 反查**，标题搜索只作 DOI 查不到时的旁路。
    """
    if not doi:
        return None
    d = http_json('https://api.crossref.org/works/' + urllib.parse.quote(doi), headers={'User-Agent': UA})
    w = d.get('message') or {}
    t = (w.get('title') or [''])[0]
    if not t:
        return None
    return {'source': 'crossref_doi', 'title': t, 'doi': w.get('DOI', ''),
            'venue': (w.get('container-title') or [''])[0],
            'year': w.get('issued', {}).get('date-parts', [[None]])[0][0],
            'cited_by': w.get('is-referenced-by-count', 0),
            'type': w.get('type', ''),
            'url': 'https://doi.org/%s' % w.get('DOI', '')}


def from_crossref_title(title):
    url = ('https://api.crossref.org/works?query.bibliographic=%s&rows=5'
           % urllib.parse.quote(title))
    d = http_json(url, headers={'User-Agent': UA})
    out = []
    for w in (d.get('message') or {}).get('items', []):
        t = (w.get('title') or [''])[0]
        if not t:
            continue
        out.append({'source': 'crossref', 'title': t, 'doi': w.get('DOI', ''),
                    'venue': (w.get('container-title') or [''])[0],
                    'year': w.get('issued', {}).get('date-parts', [[None]])[0][0],
                    'cited_by': w.get('is-referenced-by-count', 0),
                    'url': 'https://doi.org/%s' % w.get('DOI', '')})
    return out


def from_unpaywall(doi):
    """按 DOI 查 OA 位置。

    为什么用 Unpaywall 取代 OpenAlex 做 OA 定位：
    1. OpenAlex 在本机 IP 上触发硬 429（带不带 mailto 都一样），实测不可用；
    2. Unpaywall 以 DOI 为键 —— DOI 是权威标识，比标题检索更不容易错配；
    3. 它直接返回 best_oa_location.url_for_pdf，省掉「找到 OA 文章但拿不到 PDF」的空转。
    """
    if not doi:
        return None
    url = 'https://api.unpaywall.org/v2/%s?email=%s' % (urllib.parse.quote(doi), MAILTO)
    d = http_json(url, headers={'User-Agent': UA})
    loc = d.get('best_oa_location') or {}
    return {
        'source': 'unpaywall',
        'title': d.get('title') or '',
        'doi': d.get('doi') or '',
        'is_oa': d.get('is_oa', False),
        'oa_status': d.get('oa_status') or '',
        'oa_pdf': loc.get('url_for_pdf') or '',
        'oa_landing': loc.get('url_for_landing_page') or d.get('doi_landing') or '',
        'is_accepted': d.get('is_accepted', False),
        'is_published': d.get('is_published', False),
        'genre': d.get('genre') or '',
        'publisher': d.get('publisher') or '',
    }


def from_europepmc_doi(doi):
    """按 DOI 查 Europe PMC（第三个确认源，2026-10-03 新增）。

    为什么需要它：本项目大量生物力学/临床文献的元数据是从 Europe PMC 人工核验的
    （它对 1990–2020 年的医学期刊覆盖远好于 Crossref 的摘要字段）。
    但解析器原先只有 crossref_doi 与 unpaywall_title 两条确认通路，
    **Europe PMC 虽被记进 found_via、摘要也从它取，却从不参与计分** ——
    结果是「双源核验过的条目」被降为 B（单源）。

    覆盖边界：Europe PMC 不收 Physics/TPAMI/TMC 等纯工程与非医学刊，
    对这些条目它会 0 命中，**这是正常回退，不是错误**。
    """
    if not doi:
        return None
    q = urllib.parse.quote('DOI:"%s"' % doi)
    url = ('https://www.ebi.ac.uk/europepmc/webservices/rest/search'
           '?query=%s&resultType=core&format=json&pageSize=1' % q)
    d = http_json(url, headers={'User-Agent': UA})
    res = (d.get('resultList') or {}).get('result') or []
    if not res:
        return None
    r = res[0]
    jr = (r.get('journalInfo') or {}).get('journal') or {}
    return {
        'source': 'europepmc',
        'title': r.get('title') or '',
        'doi': r.get('doi') or doi,
        'pmid': r.get('pmid') or '',
        'pmcid': r.get('pmcid') or '',
        'year': (r.get('pubYear') or '')[:4] or None,
        'venue': jr.get('title') or r.get('bookOrReportDetails', {}).get('publisher') or '',
        'abstract': r.get('abstractText') or '',
        'is_oa': r.get('isOpenAccess') == 'Y',
        'in_epmc': r.get('inEPMC') == 'Y',
    }


def from_openalex_title(title):
    url = ('https://api.openalex.org/works?search=%s&per-page=5&mailto=%s'
           % (urllib.parse.quote(title), MAILTO))
    d = http_json(url)
    out = []
    for w in d.get('results', []):
        loc = w.get('primary_location') or {}
        oa = w.get('best_oa_location') or {}
        out.append({
            'source': 'openalex',
            'title': w.get('display_name') or w.get('title') or '',
            'doi': (w.get('doi') or '').replace('https://doi.org/', ''),
            'venue': ((loc.get('source') or {}) or {}).get('display_name') or '',
            'year': w.get('publication_year'), 'cited_by': w.get('cited_by_count', 0),
            'is_oa': (w.get('open_access') or {}).get('is_oa', False),
            'oa_pdf': oa.get('pdf_url') or '',
            'oa_landing': oa.get('landing_page_url') or '',
            'url': w.get('id', ''),
        })
    return out


def best_ratio(want, cand):
    return difflib.SequenceMatcher(None, norm(want), norm(cand)).ratio()


def _norm_doi(d):
    """DOI 规范化：Crossref / Europe PMC 返回的 DOI 一律小写，
    而登记的 DOI 可能是混合大小写（如 10.1016/S0031-9406(10)63103-3）。
    写回时若直接拿原始字符串做键，会匹配不上 → 该条的定级结果被静默丢弃。"""
    return (d or '').lower().strip()


def _key(p):
    """写回键：DOI 一律小写去空白，标题去首尾空白。"""
    return (_norm_doi(p.get('doi')), (p.get('arxiv_id') or '').strip(),
            (p.get('title') or '').strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    ap.add_argument('--sleep', type=float, default=0.4)
    ap.add_argument('--sources', default='crossref,unpaywall,europepmc',
                    help='逗号分隔。默认 crossref,unpaywall,europepmc（europepmc 于 2026-10-03 加入：'
                         '此前只取其摘要不计分，导致双源核验的条目被降为 B）。'
                         '加 openalex 需该源可用（实测本机 IP 会 429）。')
    args = ap.parse_args()

    data = json.loads(CAND.read_text(encoding='utf-8'))
    only = {x.strip() for x in args.only.split(',') if x.strip()}
    papers = [p for p in data['papers'] if not only or p['thread'] in only]

    # --only 必须是**合并**而非覆盖。
    # 实测踩坑：用 `--only B1` 只想省时间，结果把 94 条的全量状态覆盖成了 28 条单线程，
    # 后面 build 出来的索引就只剩 B1。局部复核不该毁掉全局工件。
    prev_by_key = {}
    if only and OUT.exists():
        try:
            _prev = json.loads(OUT.read_text(encoding='utf-8'))
            for _p in _prev.get('papers', []):
                prev_by_key[_key(_p)] = _p
            print('合并模式：已有 %d 条历史结果，本次只跑 %s' % (len(prev_by_key), ','.join(sorted(only))))
        except Exception:                                       # noqa: BLE001
            prev_by_key = {}

    results = []
    for i, p in enumerate(papers):
        rec = dict(p)
        # K38：一条畸形记录（缺 title 等必需字段）会让整轮在 100+ 条已处理后崩掉。
        # 实测：ACMM 那条 PRIOR_MANUAL 记录漏写 title，arxiv/crossref 分支直接 p['title'] → KeyError。
        # 增量落盘保住了前 105 条，但剩下 5 条要重跑 —— **不该让一条记录毁掉一轮**。
        if not rec.get('title'):
            rec['title'] = ''
            rec.setdefault('title_missing', True)
            print('[%3d/%3d] 🚨 记录缺 title，本条按「题名未知」处理（仅 DOI/arXiv 通路）'
                  % (i + 1, len(papers)), file=sys.stderr, flush=True)
        confirms, scores, all_hits = [], [], []

        if p.get('arxiv_id'):
            try:
                a = from_arxiv(p['arxiv_id'])
            except Exception as e:                              # noqa: BLE001
                a = None
                rec['arxiv_error'] = str(e)
            if a:
                all_hits.append(a)
                r = best_ratio(p['title'], a['title']) if p['title'] else 1.0
                scores.append(('arxiv', round(r, 3)))
                if not p['title'] or r >= AUTO_OK:
                    confirms.append('arxiv')
                    rec['canonical_title'] = a['title']
                    rec.setdefault('venue', a.get('journal_ref') or 'arXiv 预印本')
                    rec['year'] = rec.get('year') or (a['published'][:4] if a['published'] else None)
                    rec['doi'] = rec.get('doi') or a.get('doi', '')
                    rec['abstract'] = a.get('abstract', '')
                    rec['authors'] = a.get('authors', [])
                    rec['arxiv_published'] = a.get('published', '')
                    rec['arxiv_journal_ref'] = a.get('journal_ref', '')
                    rec['categories'] = a.get('categories', [])
            time.sleep(args.sleep)
            # arXiv 条目同样补一次 Unpaywall：拿到期刊 DOI 后定位 OA 版本
            if 'unpaywall' in args.sources and rec.get('doi'):
                try:
                    up = from_unpaywall(rec['doi'])
                except Exception:                                 # noqa: BLE001
                    up = None
                time.sleep(args.sleep)
                if up:
                    rec['is_oa'] = up.get('is_oa')
                    rec['oa_status'] = up.get('oa_status', '')
                    rec['oa_pdf'] = up.get('oa_pdf', '')
                    rec['oa_landing'] = up.get('oa_landing', '')
                    if up.get('title'):
                        scores.append(('unpaywall_title',
                                       round(best_ratio(p['title'] or a['title'], up['title']), 3)))
        else:
            # 主源：先按 DOI 直查（权威），DOI 查不到再退回标题搜索
            hit_rec, how = None, ''
            if p.get('doi'):
                try:
                    hit_rec = from_crossref_doi(p['doi'])
                except Exception as e:                          # noqa: BLE001
                    rec.setdefault('errors', []).append('crossref_doi: %r' % (e,))
                time.sleep(args.sleep)
                if hit_rec:
                    how = 'crossref_doi'
            if hit_rec is None:
                how = 'crossref_title'
                try:
                    hits = from_crossref_title(p['title'])
                except Exception as e:                          # noqa: BLE001
                    hits = []
                    rec.setdefault('errors', []).append('crossref: %r' % (e,))
                time.sleep(args.sleep)
                if hits:
                    hit_rec = max(hits, key=lambda h: best_ratio(p['title'], h['title']))
            if hit_rec:
                b = hit_rec
                r = best_ratio(p['title'], b['title'])
                scores.append((how, round(r, 3)))
                all_hits.append(b)
                if r >= AUTO_OK:
                    confirms.append(how)
                    rec['canonical_title'] = b['title']
                    rec['resolve_path'] = how
                    rec['cited_by_verified'] = b.get('cited_by')
                    rec['url_verified'] = b.get('url', '')
                    if not rec.get('venue'):
                        rec['venue'] = b.get('venue', '')
                    if not rec.get('year'):
                        rec['year'] = b.get('year')
                # 2026-10-03 新增：Europe PMC 作为独立确认源计分（此前只取摘要不计分）
                if 'europepmc' in args.sources and p.get('doi'):
                    try:
                        ep = from_europepmc_doi(p['doi'])
                    except Exception:                             # noqa: BLE001
                        ep = None
                        rec.setdefault('epmc_error', 'lookup failed')
                    time.sleep(args.sleep)
                    if ep and ep.get('title'):
                        r_ep = best_ratio(p['title'] or (hit_rec or {}).get('title', ''),
                                          ep['title'])
                        scores.append(('europepmc_doi', round(r_ep, 3)))
                        all_hits.append(ep)
                        if r_ep >= AUTO_OK:
                            confirms.append('europepmc_doi')
                            rec['epmc_pmid'] = ep.get('pmid', '')
                            rec['epmc_pmcid'] = ep.get('pmcid', '')
                            if not rec.get('abstract'):
                                rec['abstract'] = ep.get('abstract', '')
                    want_doi = (p.get('doi') or '').lower().strip()
                    got_doi = (b.get('doi') or '').lower().strip()
                    if want_doi and got_doi and want_doi != got_doi:
                        rec.setdefault('doi_conflicts', []).append(
                            {'登记DOI': want_doi, 'crossref返回DOI': got_doi,
                             'title_match': round(r, 3), 'via': how})
                    if got_doi:
                        # 只记录 Crossref 的规范形式，**不覆写** rec['doi']：
                        # 覆写会把登记的混合大小写 DOI 改成小写，导致写回键匹配失败。
                        if got_doi != (p.get('doi') or '').strip():
                            rec.setdefault('doi_canonical', got_doi)
                    # 记录类型是 peer-review / posted-content 等**不是文章本体**时，
                    # 即使标题匹配也必须降级：实测 PeerJ 的 reviews/1 页题名含论文名却 type=peer-review。
                    if b.get('type') in ('peer-review', 'component', 'grant', 'report-component'):
                        rec['doi_conflicts'].append(
                            {'登记DOI': want_doi, '登记项实为': b.get('type'),
                             '题名': b['title'][:70],
                             '说明': '登记 DOI 指向的是评审记录/组件，不是文章本体'})
                        confirms = [c for c in confirms if c != how]
                        rec['status_override'] = 'doi_conflict'
            # 副源 Unpaywall（按 Crossref 确认后的 DOI）——只做 OA 定位，不做标题裁定
            if rec.get('doi'):
                try:
                    up = from_unpaywall(rec['doi'])
                except Exception as e:                          # noqa: BLE001
                    up = None
                    rec.setdefault('errors', []).append('unpaywall: %r' % (e,))
                time.sleep(args.sleep)
                if up:
                    rec['is_oa'] = up.get('is_oa')
                    rec['oa_status'] = up.get('oa_status', '')
                    rec['oa_pdf'] = up.get('oa_pdf', '')
                    rec['oa_landing'] = up.get('oa_landing', '')
                    if up.get('is_oa') and up.get('title'):
                        scores.append(('unpaywall_title', round(best_ratio(p['title'], up['title']), 3)))
                        if best_ratio(p['title'], up['title']) >= AUTO_OK:
                            confirms.append('unpaywall')
            # 可选第三源 OpenAlex（本机 IP 已 429，默认关闭）
            if 'openalex' in args.sources:
                try:
                    hits = from_openalex_title(p['title'])
                except Exception as e:                          # noqa: BLE001
                    hits = []
                    rec.setdefault('errors', []).append('openalex: %r' % (e,))
                time.sleep(args.sleep)
                for h in hits:
                    r = best_ratio(p['title'], h['title'])
                    if r >= AUTO_OK:
                        scores.append(('openalex', round(r, 3)))
                        confirms.append('openalex')
                        break

        rec['confirm_sources'] = sorted(set(confirms))
        rec['n_confirm'] = len(set(confirms))
        rec['title_scores'] = scores
        rec['n_candidates_returned'] = len(all_hits)

        if rec.get('status_override'):
            rec['status'] = rec.pop('status_override')
            rec['evidence_grade'] = 'C（登记 DOI 指向的不是文章本体，须人工确认正确 DOI）'
        elif rec.get('doi_conflicts'):
            rec['status'] = 'doi_conflict'
            rec['evidence_grade'] = 'C（登记 DOI 与来源返回不一致，须人工裁定）'
        elif rec['n_confirm'] >= 2:
            rec['status'] = 'confirmed'
            rec['evidence_grade'] = 'A（≥2 个独立来源确认标题）'
        elif rec['n_confirm'] == 1 and max((s for _, s in scores), default=0) >= AUTO_OK:
            rec['status'] = 'confirmed_single'
            rec['evidence_grade'] = 'B（单来源确认标题，>=0.90）'
        elif max((s for _, s in scores), default=0) >= NEED_REVIEW:
            rec['status'] = 'review'
            rec['evidence_grade'] = 'C（相似度 0.74-0.90，须人工复核）'
        else:
            rec['status'] = 'unresolved'
            rec['evidence_grade'] = 'D（未达阈值或无来源命中，不得引用）'
        results.append(rec)
        print('[%3d/%3d] %-16s %-3d源 %-5s %s' % (
            i + 1, len(papers), rec['status'], rec['n_confirm'],
            (p.get('arxiv_id') or '-')[:10], (rec.get('canonical_title') or p['title'])[:52]), flush=True)
        # 增量落盘 + 合并历史（保持候选清单的原始顺序，避免局部跑打乱全量）
        merged = {}
        for _p in results:
            merged[_key(_p)] = _p
        final = []
        for _p in data['papers']:
            k = _key(_p)
            final.append(merged.get(k) or prev_by_key.get(k) or _p)
        # 落地校验：处理过的条目必须真的落进产物，否则是静默数据丢失（K31）
        landed = {_norm_doi(x.get('doi')) for x in final}
        lost = [r for r in results
                if _norm_doi(r.get('doi')) not in landed or 'status' not in
                next((f for f in final
                      if _norm_doi(f.get('doi')) == _norm_doi(r.get('doi'))), {})]
        if lost:
            print('🚨 写回丢失 %d 条：%s' % (
                len(lost), ', '.join((r.get('canonical_title') or r.get('title') or '?')[:40]
                                     for r in lost[:5])), file=sys.stderr)
        OUT.write_text(json.dumps({'meta': data['meta'], 'papers': final}, ensure_ascii=False, indent=2),
                       encoding='utf-8')

    if not results:
        print('没有任何条目匹配 --only %s，未改动产物。' % (sorted(only) or ['<空>']),
              file=sys.stderr)
        return

    OUT.write_text(json.dumps({'meta': data['meta'], 'papers': final}, ensure_ascii=False, indent=2),
                   encoding='utf-8')
    c = Counter(r['status'] for r in results)
    g = Counter(r['evidence_grade'][0] for r in results)
    print('\n汇总：%s' % dict(c), file=sys.stderr)
    print('证据等级分布：%s' % dict(g), file=sys.stderr)


if __name__ == '__main__':
    main()
