#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rescue.py — 对「未命中」条目做最后一轮抢救检索，把可救的找回来。

背景：首轮 ti:"整标题" 检索对标题含冒号、或我记忆中的标题与 arXiv 收录写法有出入的条目会落空。
本脚本改用**局部片段**检索（取标题中最具辨识度的连续词组），再按整体相似度排序，
把候选连同 arXiv ID 一并输出，**由人工逐条裁定**（不自动采信）。

用法：python3 rescue.py
"""
import difflib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESOLVED = HERE / 'corpus_resolved.json'
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36'
API = 'https://export.arxiv.org/api/query'

# 明确不在 arXiv 的条目：直接排除，不必浪费检索
KNOWN_NOT_ARXIV = {
    'Language Models are Unsupervised Multitask Learners',      # GPT-2，仅 openai.com 技术报告
    'A BERT Based Method for Continuous Estimation of Cross-Subject Hand Kinematics From Surface Electromyographic Signals',
    'A Dual-Task Improved Transformer Framework for Decoding Lower Limb Sit-to-Stand Movement from sEMG and IMU Data',
    'Markerless On-Device Detection of Compensatory Movement Patterns in Upper-Limb Rehabilitation Exercises from Monocular RGB Video',
    'Using deep learning to detect upper limb compensation in individuals post-stroke using consumer-grade webcams',
    'Prediction of Joint Angles for Human Elbow Motion Based on sEMG',
    'Video generation models as world simulators',              # Sora 报告，未上 arXiv
}


def norm(s):
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).split())


def query(term, n=8):
    url = '%s?search_query=%s&max_results=%d' % (API, urllib.parse.quote('ti:"%s"' % term), n)
    xml = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}),
                                 timeout=40).read().decode('utf-8', 'replace')
    out = []
    for blk in re.findall(r'<entry>(.*?)</entry>', xml, re.S):
        aid = re.search(r'<id>http://arxiv\.org/abs/([^<]+)</id>', blk)
        t = re.search(r'<title>(.*?)</title>', blk, re.S)
        if aid and t:
            out.append((aid.group(1), ' '.join(re.sub(r'<[^>]+>', '', t.group(1))
                                               .replace('&amp;', '&').split())))
    return out


def fragments(title):
    """从标题里取若干有辨识度的片段做检索。"""
    t = re.sub(r'[^A-Za-z0-9 ]+', ' ', title)
    w = t.split()
    frs = []
    for n in (7, 6, 5):
        if len(w) >= n:
            frs.append(' '.join(w[:n]))       # 开头
            frs.append(' '.join(w[-n:]))      # 结尾（arXiv 常把方法名放最前，标题后段才是正文描述）
    seen, out = set(), []
    for f in frs:
        if f and f not in seen:
            seen.add(f); out.append(f)
    return out


def main():
    d = json.loads(RESOLVED.read_text(encoding='utf-8'))
    todo = [p for p in d['papers'] if p.get('status') != 'confirmed' and p.get('title') not in KNOWN_NOT_ARXIV]
    print('待抢救条目：%d（另有 %d 条已确认不在 arXiv，直接进人工清单）\n' % (len(todo), len(KNOWN_NOT_ARXIV)))

    rescued = {}
    for p in todo:
        want = p.get('title', '')
        best = (0.0, None, None, '')
        for frag in fragments(want):
            try:
                hits = query(frag)
            except Exception:                        # noqa: BLE001
                time.sleep(4); continue
            time.sleep(3.3)
            for aid, t in hits:
                r = difflib.SequenceMatcher(None, norm(want), norm(t)).ratio()
                if r > best[0]:
                    best = (r, aid, t, frag)
        if best[1] and best[0] >= 0.72:
            rescued[want] = {'arxiv_id': best[1], 'canonical_title': best[2],
                             'score': round(best[0], 3), 'via': best[3]}
            print('  ?  %.2f  %-46s -> %-13s %s' % (best[0], want[:44], best[1], best[2][:52]))
        else:
            print('  ✗  %.2f  %-46s 仍无可靠命中' % (best[0], want[:44]))

    (HERE / 'rescue_candidates.json').write_text(
        json.dumps(rescued, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n可救候选 %d 条已写入 rescue_candidates.json —— **需人工逐条确认后才写入语料**' % len(rescued))


if __name__ == '__main__':
    main()
