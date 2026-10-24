#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_oa.py — 取回语料二「人工取件清单」里**开放获取（OA）**的条目。

纪律（与项目既有规范一致）：
  - 只走 **官方 OA 位置**（Unpaywall 给出的出版商直链 / CVF / AAAI OJS）或 **开放获取仓库**。
  - **不使用影子图书馆、不规避付费墙、不绕过反爬验证**。
  - 每份下载都过三重校验：`%PDF` magic + 体积下限 + `file` 判定；
    出版商返回反爬验证页/HTML 时**如实判失败**，不伪装成功。
  - 校验不通过的条目写入 `fetch_oa_report.json`，由人决定是否用浏览器人工下载。

用法：python3 fetch_oa.py [--only 子串]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TF = (HERE / '../../01_参考论文/运动理解/Transformer_发展线').resolve()
REPORT = HERE / 'fetch_oa_report.json'
UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/124.0 Safari/537.36')
MIN_BYTES = 20 * 1024

# (slug, 论文短名, 线程, 目标文件名, [候选 URL], 备注)
TARGETS = [
    ("machines13100953", "iTransformer-DTL 坐立起身 sEMG+IMU 解码", "T5",
     "2025_iTransformer-DTL_坐立起身_sEMG_IMU_轨迹解码.pdf",
     # www.mdpi.com 对脚本返回 403（反爬）；res.mdpi.com 是同一出版商的官方 PDF CDN，
     # 属正常官方 OA 通道，不是绕过。先后顺序：CDN 优先。
     ["https://res.mdpi.com/d_attachment/machines/machines-13-00953/article_deploy/machines-13-00953.pdf",
      "https://www.mdpi.com/2075-1702/13/10/953/pdf"],
     "MDPI Machines, Gold OA CC-BY, DOI 10.3390/machines13100953"),
    ("s26165054", "端侧单目 RGB 上肢代偿检测", "T6",
     "2026_端侧单目RGB_上肢代偿模式检测.pdf",
     ["https://res.mdpi.com/d_attachment/sensors/sensors-26-05054/article_deploy/sensors-26-05054.pdf",
      "https://www.mdpi.com/1424-8220/26/16/5054/pdf"],
     "MDPI Sensors, Gold OA CC-BY, DOI 10.3390/s26165054"),
    ("fmed2025", "消费级摄像头跨被试上肢代偿检测", "T6",
     "2025_消费级摄像头_跨被试上肢代偿检测.pdf",
     ["https://public-pages-files-2025.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1645369/pdf",
      "https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1645369/pdf"],
     "Frontiers in Medicine, Gold OA CC-BY, DOI 10.3389/fmed.2025.1645369"),
    ("aaai42465", "Pre-Trained Video Generative Models as World Simulators", "T6",
     "2026_AAAI_预训练视频生成模型作为世界模拟器.pdf",
     ["https://ojs.aaai.org/index.php/AAAI/article/download/42465/46426"],
     "AAAI 2026, Gold OA, DOI 10.1609/aaai.v40i6.42465（替代项，非 Sora 报告本身）"),
]


def is_pdf(path: Path) -> tuple[bool, str]:
    try:
        n = path.stat().st_size
    except OSError as e:
        return False, str(e)
    if n < MIN_BYTES:
        return False, '仅 %d B，疑似验证页' % n
    with open(path, 'rb') as f:
        if f.read(5) != b'%PDF-':
            head = path.read_bytes()[:200].decode('utf-8', 'ignore')
            kind = '反爬验证页' if re.search(r'captcha|cloudflare|verify you are human|access denied',
                                            head, re.I) else '非 PDF（HTML）'
            return False, kind
    r = subprocess.run(['file', '-b', str(path)], capture_output=True, text=True)
    if 'PDF document' not in r.stdout:
        return False, 'file 判定非 PDF: %s' % r.stdout.strip()[:50]
    return True, ''


def try_download(url: str, dest: Path) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/pdf,*/*'})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = r.read()
        dest.write_bytes(data)
        ok, why = is_pdf(dest)
        if ok:
            return True, url
        dest.unlink(missing_ok=True)
        return False, '%s ← %s' % (why, url)
    except Exception as e:                        # noqa: BLE001
        return False, '%r ← %s' % (str(e)[:90], url)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    args = ap.parse_args()
    rows = [t for t in TARGETS if not args.only or args.only in t[1] or args.only in t[0]]

    results = []
    for slug, label, thread, fname, urls, note in rows:
        outdir = TF / ('%s_%s' % (thread, {'T5': '动作生成与离散token', 'T6': '物理约束·世界模型'}[thread]))
        outdir.mkdir(parents=True, exist_ok=True)
        dest = outdir / fname
        if dest.exists():
            ok, why = is_pdf(dest)
            if ok:
                results.append({'slug': slug, 'label': label, 'status': 'ok', 'path': str(dest.relative_to(TF)),
                                'bytes': dest.stat().st_size, 'note': note, 'url': '(已存在)'})
                print('  已有  %-46s %s' % (label[:44], fname[:52]))
                continue
        errs = []
        got = None
        for u in urls:
            ok, why = try_download(u, dest)
            if ok:
                got = u
                break
            errs.append(why)
            time.sleep(2)
        if got:
            results.append({'slug': slug, 'label': label, 'status': 'ok', 'path': str(dest.relative_to(TF)),
                            'bytes': dest.stat().st_size, 'note': note, 'url': got})
            print('  ✓     %-46s %6d B' % (label[:44], dest.stat().st_size))
        else:
            results.append({'slug': slug, 'label': label, 'status': 'failed', 'path': '',
                            'bytes': 0, 'note': note, 'url': ' | '.join(urls), 'errors': errs})
            print('  ✗     %-46s %s' % (label[:44], errs[0][:70] if errs else ''))
        REPORT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    ok = sum(1 for r in results if r['status'] == 'ok')
    print('\nOA 取回 %d/%d' % (ok, len(results)))
    for r in results:
        if r['status'] != 'ok':
            print('  需人工（浏览器下载）: %s' % r['label'])
    return 0 if ok == len(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
