#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adjudicate.py — 人工裁定：核对「我记忆中的 arXiv ID ↔ 论文标题」是否属实。

为什么需要：
  首轮检索对部分条目返回低相似度或空结果，两种可能：
    (a) 该论文确实不在 arXiv（期刊论文 / 企业技术报告）→ 应进人工取件清单
    (b) 我写的标题与 arXiv 收录标题不一致 → 应改用 arXiv 的规范标题
  本脚本的做法是**由我给出 ID 假设，再由 arXiv 官方 API 反查真实标题**：
  标题相似度达标 → 证实我的记忆，改用规范标题；
  相似度不达标或查无此 ID → 证伪，该条目不得作为证据使用。

用法：python3 adjudicate.py
"""
import difflib
import json
import re
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESOLVED = HERE / 'corpus_resolved.json'
CAND = HERE / 'corpus_candidates.json'
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36'

# 我记忆中的 arXiv ID → 期望标题（脚本用 API 反查验证，验证不通过则丢弃）
HYPOTHESES = {
    '1904.10509': 'Generating Long Sequences with Sparse Transformers',
    '2205.14135': 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness',
    '2211.17192': 'Accelerating Large Language Model Inference with Speculative Decoding',
    '2003.02407': 'Self-supervised learning for human activity recognition using wearable sensors',
    '2304.08485': 'LLaVA: Visual Instruction Tuning',
    '2212.06983': 'InternVideo: Generalizing Video Foundation Models to General Modalities',
    '2307.08621': 'Retentive Network: A Successor to Transformer for Large Language Models',
    '2412.04045': 'Byte Latent Transformer: Patches Scale the Inference Parameters of LLMs',
    '2502.11089': 'Native Sparse Attention',
    '2009.14794': 'Performer: Rethinking Attention with Performers',
    '2310.16828': 'TD-MPC2: Scalable, Robust World Models with Continuous Action Spaces',
}
THRESH = 0.80


def norm(s):
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).split())


def fetch(aid):
    url = 'https://export.arxiv.org/api/query?id_list=%s' % aid
    xml = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}),
                                 timeout=40).read().decode('utf-8', 'replace')
    if '<entry>' not in xml:
        return None
    blk = re.search(r'<entry>(.*?)</entry>', xml, re.S).group(1)

    def g(tag, src=blk):
        m = re.search(r'<%s[^>]*>(.*?)</%s>' % (tag, tag), src, re.S)
        if not m:
            return ''
        v = re.sub(r'<[^>]+>', '', m.group(1))
        return ' '.join(v.replace('&amp;', '&').replace('&lt;', '<')
                        .replace('&gt;', '>').replace('&quot;', '"').split())
    return {
        'arxiv_id': g('id').rsplit('/abs/', 1)[-1],
        'title': g('title'),
        'authors': re.findall(r'<name>(.*?)</name>', blk),
        'published': g('published'), 'updated': g('updated'),
        'primary_category': (re.findall(r'<category term="([^"]+)"', blk) or [''])[0],
        'categories': re.findall(r'<category term="([^"]+)"', blk),
        'journal_ref': g('arxiv:journal_ref'), 'comment': g('arxiv:comment'),
        'doi': g('arxiv:doi'), 'abstract': g('summary'),
    }


def main():
    confirmed, rejected = {}, []
    for aid, want in HYPOTHESES.items():
        try:
            e = fetch(aid)
        except Exception as ex:                        # noqa: BLE001
            print('  ERR  %s  %r' % (aid, ex)); rejected.append((aid, '请求失败')); time.sleep(3.2); continue
        if e is None:
            print('  ✗    %s  arXiv 查无此 ID' % aid); rejected.append((aid, '查无此 ID')); time.sleep(3.2); continue
        r = difflib.SequenceMatcher(None, norm(want), norm(e['title'])).ratio()
        if r >= THRESH:
            e['match_score'] = round(r, 3)
            e['query'] = 'id_list 人工反查证实'
            confirmed[aid] = e
            print('  ✓    %s  %.2f  %s' % (aid, r, e['title'][:64]))
        else:
            print('  ✗    %s  相似度仅 %.2f：期望「%s」实为「%s」' % (aid, r, want[:34], e['title'][:44]))
            rejected.append((aid, '标题不符：实为 %s' % e['title'][:50]))
        time.sleep(3.2)

    # 用证实结果修补 corpus_candidates.json（改用 arXiv 规范标题）
    cand = json.loads(CAND.read_text(encoding='utf-8'))
    patched = 0
    for i, p in enumerate(cand['papers']):
        aid = (p.get('arxiv_id') or '').split('v')[0]
        hit = None
        for k, e in confirmed.items():
            r = difflib.SequenceMatcher(None, norm(p['title']), norm(e['title'])).ratio()
            if r >= 0.62:                              # 旧写法与规范标题的近似度
                hit = e; break
        if hit:
            p['title'] = hit['title']
            p['arxiv_id'] = hit['arxiv_id']
            p['doi'] = p.get('doi') or hit.get('doi', '')
            patched += 1
    CAND.write_text(json.dumps(cand, ensure_ascii=False, indent=2), encoding='utf-8')

    # 直接写入已核验结果（不再走检索，避免二次误配）
    res = json.loads(RESOLVED.read_text(encoding='utf-8'))
    n = 0
    for rec in res['papers']:
        if rec.get('status') in ('confirmed',):
            continue
        cand_hit = None
        for k, e in confirmed.items():
            r = difflib.SequenceMatcher(None, norm(rec.get('title', '')), norm(e['title'])).ratio()
            if r >= 0.62:
                cand_hit = e; break
        if not cand_hit:
            continue
        for k, v in rec.items():
            if k in ('status', 'match_score', 'arxiv_id', 'candidates', 'error', 'reason'):
                continue
            rec[k] = v
        rec.update({
            'status': 'confirmed',
            'match_score': cand_hit['match_score'],
            'arxiv_id': cand_hit['arxiv_id'],
            'canonical_title': cand_hit['title'],
            'authors': cand_hit['authors'],
            'published': cand_hit['published'],
            'updated': cand_hit['updated'],
            'primary_category': cand_hit['primary_category'],
            'categories': cand_hit['categories'],
            'journal_ref': cand_hit['journal_ref'],
            'comment': cand_hit['comment'],
            'doi': cand_hit['doi'],
            'abstract': cand_hit['abstract'],
            'api_year': int(cand_hit['published'][:4]) if cand_hit['published'] else rec.get('year'),
            'query': 'id_list 人工反查证实',
        })
        rec['year_mismatch'] = (rec['api_year'] != rec.get('year'))
        n += 1
    RESOLVED.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n证实 %d 个 ID，修正候选标题 %d 条，判定通过 %d 条' % (len(confirmed), patched, n))
    if rejected:
        print('证伪/无法证实（不得作为证据，须进人工取件清单）：')
        for a, why in rejected:
            print('  · %s  %s' % (a, why))


if __name__ == '__main__':
    main()
