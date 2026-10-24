#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_arxiv.py — 按 arXiv ID 取开放获取 PDF（三重校验），用于**优先件**。

与语料二 download_pdfs.py 的区别：
    download_pdfs.py 消费 corpus_resolved.json（批量、语料级）。
    本脚本消费显式 ID 列表（少量、优先级件），用于在正式建链之前先把
    「必须先读」的几篇拿到手，并留下与语料二同规格的机器报告。

三重校验（防止把 HTML 错误页当 PDF 存入库）：
    1. 文件头必须是 %PDF
    2. 体积下限（默认 40 KB）
    3. file(1) 判定为 PDF 文档

用法：
    python3 fetch_arxiv.py --dest ../01_参考论文/整合系统/生物力学线/_优先件 \
        --ids 2402.17192,2411.14992,2603.26844,2609.18667,2607.23608
"""
import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
SLEEP = 3.2          # arXiv 官方建议 ≥3 秒/请求
MIN_BYTES = 40 * 1024


def validate(path):
    """三重校验，返回 (ok, reason, bytes)"""
    try:
        size = path.stat().st_size
    except OSError as e:
        return False, 'stat 失败 %r' % (e,), 0
    if size < MIN_BYTES:
        return False, '体积过小 %d B' % size, size
    with open(path, 'rb') as f:
        if f.read(4) != b'%PDF':
            return False, '文件头不是 %%PDF', size
    try:
        out = subprocess.run(['file', '-b', str(path)], capture_output=True, text=True, timeout=20).stdout
    except Exception as e:                                       # noqa: BLE001
        return False, 'file(1) 不可用 %r' % (e,), size
    if 'PDF document' not in out:
        return False, 'file(1) 判定非 PDF: %s' % out.strip()[:60], size
    return True, 'ok', size


def fetch(aid, dest, retries=3):
    dest.mkdir(parents=True, exist_ok=True)
    url = 'https://arxiv.org/pdf/%s' % aid
    tmp = dest / (aid + '.part')
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            tmp.write_bytes(data)
            ok, reason, size = validate(tmp)
            if ok:
                final = dest / ('%s.pdf' % aid)
                tmp.rename(final)
                return {'arxiv_id': aid, 'status': 'ok', 'path': str(final.relative_to(dest.parent)),
                        'bytes': size, 'url': url, 'reason': reason}
            last = reason
        except Exception as e:                                   # noqa: BLE001
            last = repr(e)
        time.sleep(4 * (attempt + 1))
    if tmp.exists():
        tmp.unlink()
    return {'arxiv_id': aid, 'status': 'failed', 'bytes': 0, 'url': url, 'reason': str(last)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', required=True)
    ap.add_argument('--dest', required=True)
    ap.add_argument('--report', default=str(HERE / 'fetch_priority_report.json'))
    args = ap.parse_args()

    dest = Path(args.dest).resolve()
    ids = [x.strip() for x in args.ids.split(',') if x.strip()]
    items = []
    for i, aid in enumerate(ids):
        r = fetch(aid, dest)
        items.append(r)
        print('[%d/%d] %-6s %-12s %s' % (i + 1, len(ids), r['status'].upper(), aid,
                                         ('%d B' % r['bytes']) if r['status'] == 'ok' else r['reason'][:60]),
              flush=True)
        time.sleep(SLEEP)

    ok = sum(1 for r in items if r['status'] == 'ok')
    Path(args.report).write_text(json.dumps(
        {'dest': str(dest), 'requested': len(ids), 'ok': ok, 'failed': len(ids) - ok, 'items': items},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n完成：%d/%d 成功 → %s' % (ok, len(ids), dest))


if __name__ == '__main__':
    main()
