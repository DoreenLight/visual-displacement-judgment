#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_pdfs.py — 按线程批量下载 arXiv 开放获取 PDF，并生成「需人工补件清单」。

原则：
  - 只取 arXiv 官方开放获取 PDF（合法 OA），不做任何付费墙绕过。
  - 每篇下载后校验：%PDF magic + 体积下限 + `file` 合法 PDF。
  - 失败项不静默丢弃，全部写入 corpus_download_report.json 的 needs_manual，
    该清单即交由用户通过学校数据库人工补件。

用法：
    python3 download_pdfs.py                 # 只下尚未存在的
    python3 download_pdfs.py --retry-failed  # 重试失败项
    python3 download_pdfs.py --dest ../../01_参考论文/运动理解/Transformer_发展线
"""
import argparse
import json
import re
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESOLVED = HERE / 'corpus_resolved.json'
REPORT = HERE / 'corpus_download_report.json'

UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
SLEEP = 1.6          # PDF 下载比 API 查询宽松；失败项有重试 + 合法性校验兜底
MIN_BYTES = 20 * 1024


def slugify(title, year, authors):
    a = (authors[0].split()[-1] if authors else 'anon')
    a = re.sub(r'[^A-Za-z]', '', a) or 'anon'
    body = re.sub(r'[^A-Za-z0-9]+', '_', title).strip('_')
    words = body.split('_')
    short = '_'.join(words[:7]) if len(words) > 7 else body
    if len(short) > 60:
        short = short[:60].rstrip('_')
    return '%s_%s_%s' % (year, a, short)


def is_valid_pdf(path):
    try:
        if path.stat().st_size < MIN_BYTES:
            return False, '文件过小 %d B' % path.stat().st_size
        with open(path, 'rb') as f:
            if f.read(5) != b'%PDF-':
                return False, '非 PDF magic（可能是 HTML 错误页）'
    except Exception as e:                            # noqa: BLE001
        return False, str(e)
    r = subprocess.run(['file', '-b', str(path)], capture_output=True, text=True)
    if 'PDF document' not in r.stdout:
        return False, 'file 判定非 PDF: %s' % r.stdout.strip()[:60]
    return True, ''


def download(url, dest, retries=3):
    last = ''
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            dest.write_bytes(data)
            ok, why = is_valid_pdf(dest)
            if ok:
                return True, ''
            last = why
            dest.unlink(missing_ok=True)
        except Exception as e:                        # noqa: BLE001
            last = repr(e)[:120]
        time.sleep(5 * (attempt + 1))
    return False, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default='../../01_参考论文/运动理解/Transformer_发展线')
    ap.add_argument('--retry-failed', action='store_true')
    args = ap.parse_args()

    dest_root = (HERE / args.dest).resolve()
    data = json.loads(RESOLVED.read_text(encoding='utf-8'))
    threads = data.get('meta', {}).get('axes', {}).get('thread', {})
    papers = [p for p in data['papers'] if p.get('status') in ('confirmed', 'review') and p.get('arxiv_id')]

    prev = {}
    if REPORT.exists():
        try:
            prev = {r['arxiv_id']: r for r in json.loads(REPORT.read_text(encoding='utf-8'))['items']}
        except Exception:                            # noqa: BLE001
            prev = {}

    items = []
    for n, p in enumerate(papers, 1):
        aid = p['arxiv_id']
        th = p['thread']
        # 目录名必须与 build_index.py / gen_cards.py 完全一致：<线程号>_<线程名>
        thdir = ('%s_%s' % (th, threads.get(th, th))).replace(' ', '')
        outdir = dest_root / thdir
        outdir.mkdir(parents=True, exist_ok=True)
        fname = slugify(p.get('canonical_title') or p['title'], p.get('api_year') or p.get('year'), p.get('authors', [])) + '.pdf'
        target = outdir / fname
        rel = str(target.relative_to(dest_root.parent.parent))

        if target.exists():
            ok, why = is_valid_pdf(target)
            if ok:
                items.append({'arxiv_id': aid, 'status': 'ok', 'path': rel,
                              'bytes': target.stat().st_size, 'reason': 'already'})
                print('[%3d/%d] 已有  %-9s %s' % (n, len(papers), aid, fname[:56]), flush=True)
                continue

        if not args.retry_failed and aid in prev and prev[aid].get('status') == 'failed':
            items.append(prev[aid])
            print('[%3d/%d] 跳过(失败已记录) %-9s %s' % (n, len(papers), aid, fname[:40]), flush=True)
            continue

        ok, why = download('https://arxiv.org/pdf/%s' % aid, target)
        rec = {'arxiv_id': aid, 'status': 'ok' if ok else 'failed', 'path': rel if ok else '',
               'bytes': target.stat().st_size if ok else 0, 'reason': why,
               'title': p.get('canonical_title') or p['title'], 'thread': th,
               'stage': p.get('stage'), 'year': p.get('api_year') or p.get('year')}
        items.append(rec)
        print('[%3d/%d] %s  %-9s %s %s' % (n, len(papers), 'OK   ' if ok else 'FAIL ', aid,
                                           fname[:48], ('' if ok else '← ' + why)), flush=True)
        REPORT.write_text(json.dumps({'items': items}, ensure_ascii=False, indent=2), encoding='utf-8')
        time.sleep(SLEEP)

    REPORT.write_text(json.dumps({'items': items}, ensure_ascii=False, indent=2), encoding='utf-8')
    c = Counter(i['status'] for i in items)
    nf = [i for i in items if i['status'] != 'ok']
    print('\n下载汇总：%s ｜ 需人工补件 %d 篇' % (dict(c), len(nf)), flush=True)
    for i in nf:
        print('  · %s  %s' % (i['arxiv_id'], (i.get('title') or '')[:60]), flush=True)


if __name__ == '__main__':
    main()
