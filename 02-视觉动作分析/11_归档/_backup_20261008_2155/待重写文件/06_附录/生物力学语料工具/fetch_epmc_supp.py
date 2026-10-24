#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_epmc_supp.py — 取回 Europe PMC 收录文章的**补充材料**（supplementary files）

为什么需要这个工具
------------------
2026-10-02 的一次误判逼出来的。当时 `recover_oa.py` 判定 Pohl 等 2026
（`10.2196/87339`）的补充表「须人工取件」，理由是：
  ① Europe PMC 的 JATS 正文 **不含** supplementary —— 这条是**对的**
  ② `preprints.jmir.org` 直连返回 HTTP 202 空响应（反爬）—— 这条是**对的**
两条都对，但**都不是结论**：真正的通路当时没试。

Europe PMC 有一条**独立于全文接口**的补充材料接口：

    GET https://www.ebi.ac.uk/europepmc/webservices/rest/<PMCID>/supplementaryFiles
    → 200 application/zip（实测 PMC13293569 = 805,093 bytes / 4 PDF + 3 webp）

同期的另一条官方通道 `ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=<PMCID>`
对同一 PMCID 返回 **404**（非反爬）。**两条通道结论可以不一致，都要记。**

🚨 取件时最容易踩的坑（也是当初误判的直接原因）
------------------------------------------------
    **文件名编号 ≠ 表号编号。**
    Pohl 那篇正文引用的「补充表 S2 / S3」全部在 `s002.pdf` **一份文件里**；
    打开会看到表号从 S1 开始。
    → 按「文件名 = 表号」的直觉去找，很容易误判成「这不是我要的材料」并就此收手。
    → **本工具默认会为每份下载件生成一份 `_清单.txt`，列出 PDF 的页数与首行文字**，
      让人在**不打开每个文件**的前提下先确认自己拿到的是不是要的东西。

纪律
----
  · 只走 Europe PMC / NCBI 的公开接口，**不绕过任何反爬、不碰付费墙**
  · 只做**探测 + 取件**，不做内容判断 —— 「这份补充材料证明了什么」由人决定
  · 404 / 空响应**如实记录**，不静默跳过（K11 同源：零命中要有解释）
  · 🚨 **HTTP 200 ≠ 有补充材料**（实测教训，见下）
  · 取件**不改语料口径**：补充材料是已入库论文的附件、不是独立论文条目，
    落盘位置由调用方指定，默认写到 `06_附录/补充材料/<PMCID>/`（**不进语料三**）

🚨🚨 最重要的实测教训：**这个接口会把正文插图一起打包**
------------------------------------------------------
接口名叫 `supplementaryFiles`，但对**每一个** PMCID 都返回 `HTTP 200 + zip`，
包括**根本没有补充材料**的条目 —— 因为它把文章的**配图**（`g00N.gif`、`FigN_HTML.jpg`）
和公式图（`Article_IEqN.gif`）一并塞进去了。

    实测：语料三 12 个 PMCID → 12 个都是 HTTP 200
          但**真补充件只有 5 个**，另外 7 个包里只有正文插图。

**因此 `--probe` 的「有补充材料」判定必须按文件名分类，不能只看 HTTP 状态。**
本工具用 `CLASSIFY_RE` 区分：

    真补充件 : ``-s00N.`` / ``MOESM*`` / ``*_ESM.`` / ``Data_Sheet*`` / ``Table_*.DOCX``
    正文插图 : ``-g00N.`` / ``*_FigN_HTML.`` / ``*_Article_[IE]EqN.``

**首次使用时打印的「有补充材料」是按 HTTP 判的，会高估；**
`--classify` 才是按文件名判的准数。**两个数都要看。**

用法
----
    python3 fetch_epmc_supp.py --pmcid PMC13293569
    python3 fetch_epmc_supp.py --probe                 # 只探测不下载
    python3 fetch_epmc_supp.py --from-corpus           # 对语料三全部 XML 逐个探测
    python3 fetch_epmc_supp.py --pmcid PMCxxxx --out <目录>
"""
import argparse
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

EPMC = 'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles'
# 正文插图（接口会混装进来，**不是补充材料**）
FIG_RE = re.compile(r'(-g\d+\.(gif|jpg|jpeg|png|webp)$)|(_Fig\d+.*_HTML\.)|(_Article_[IE]?Eq\d+\.)', re.I)
# 真补充件的命名特征
SUPP_RE = re.compile(r'(-s\d+\.(pdf|zip|docx?|xlsx?)$)|(MOESM)|(_ESM\.)|(Data_Sheet)|(^Table_\d+\.)', re.I)


def classify(name):
    """返回 'supp' | 'fig'。**按文件名判，不看 HTTP 状态**（接口会把插图混装进来）。"""
    return 'fig' if FIG_RE.search(name) else ('supp' if SUPP_RE.search(name) else 'other')
NCBI_OA = 'https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}'
UA = 'kb-fetch-epmc-supp/1.0 (local research; contact: local)'


def http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read(), dict(r.headers)


def probe(pmcid, timeout=60):
    """只问「有没有补充材料」，不下载。返回 (有?, 详情dict)。"""
    out = {'pmcid': pmcid}
    try:
        st, body, hdrs = http_get(EPMC.format(pmcid=pmcid), timeout)
        ctype = hdrs.get('Content-Type', '')
        out['epmc_status'] = st
        out['epmc_bytes'] = len(body)
        out['epmc_type'] = ctype
        if st == 200 and body[:2] == b'PK':
            out['has_supplementary'] = True
            try:
                with zipfile.ZipFile(io.BytesIO(body)) as z:
                    out['members'] = z.namelist()
            except zipfile.BadZipFile:
                out['has_supplementary'] = False
                out['note'] = '返回 200 但不是 zip，判为无补充材料'
        else:
            out['has_supplementary'] = False
    except urllib.error.HTTPError as e:
        out['epmc_status'] = e.code
        out['has_supplementary'] = False
        out['note'] = f'HTTPError {e.code}（不是反爬就是没有，**不要自行绕过**）'
    except Exception as e:                                    # noqa: BLE001
        out['has_supplementary'] = False
        out['note'] = f'{type(e).__name__}: {e}'
    # 第二条官方通道：结论可能不同，**两条都记**（见文件头）
    try:
        st2, b2, _ = http_get(NCBI_OA.format(pmcid=pmcid), timeout=30)
        out['ncbi_oa_status'] = st2
    except urllib.error.HTTPError as e:
        out['ncbi_oa_status'] = e.code
    except Exception:                                           # noqa: BLE001
        out['ncbi_oa_status'] = None
    return out


def fetch(pmcid, outdir, timeout=120):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    st, body, _ = http_get(EPMC.format(pmcid=pmcid), timeout)
    if st != 200 or body[:2] != b'PK':
        raise RuntimeError(f'{pmcid}: HTTP {st}，非 zip，未落盘')
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        z.extractall(outdir)
        names = z.namelist()
    # 生成清单：让人不用逐个打开就能确认拿到的是不是要的材料
    supp = [n for n in names if classify(n) == 'supp']
    figs = [n for n in names if classify(n) == 'fig']
    other = [n for n in names if classify(n) == 'other']
    lines = [f'# {pmcid} 补充材料取件清单', '',
             f'- ⚠️ **HTTP 200 不等于有补充材料** —— 本接口把正文插图一并打包。',
             f'- **按文件名分类：真补充件 {len(supp)} 个 / 正文插图 {len(figs)} 个 / 未分类 {len(other)} 个**', '',
             '## ✅ 真补充件（这些才是要逐份打开核对的）', ''] + \
            [f'- `{n}`' for n in sorted(supp)] + ['',
             '## 附：正文插图（XML 已有，**不必重复保存**，此处仅列出以示区分）', ''] + \
            [f'- `{n}`' for n in sorted(figs)] + (['- 未分类：' + ', '.join(sorted(other))] if other else []) + ['',
             f'- 取件时间：{time.strftime("%Y-%m-%d %H:%M:%S")}',
             f'- 接口：`GET {EPMC.format(pmcid=pmcid)}`',
             f'- HTTP 200 / {len(body):,} bytes / {len(names)} 个文件', '',
             '## 文件', '']
    for n in sorted(names):
        fp = outdir / n
        sz = fp.stat().st_size if fp.exists() else 0
        first = ''
        if n.lower().endswith('.pdf'):
            try:
                import subprocess
                t = subprocess.run(['pdftotext', '-l', '1', '-layout', str(fp), '-'],
                                   capture_output=True, timeout=60).stdout.decode('utf-8', 'ignore')
                first = ' '.join(t.split())[:220]
            except Exception:                                   # noqa: BLE001
                first = '(pdftotext 不可用或解析失败)'
        lines.append(f'- `{n}`  {sz:,} bytes')
        if first:
            lines.append(f'  - 首行：{first}')
    lines += ['', '## 取件信息', '',
              f'- 取件时间：{time.strftime("%Y-%m-%d %H:%M:%S")}',
              f'- 接口：`GET {EPMC.format(pmcid=pmcid)}`',
              f'- HTTP 200 / {len(body):,} bytes / {len(names)} 个文件', '',
              '## ⚠️ 取件纪律提醒', '',
              '- **文件名编号 ≠ 表号编号**：本项目 Pohl 2026 那篇，正文引用的「表 S2/S3」'
              '全部在 `s002.pdf` 一份里。取件后**必须逐份打开核对表号**，不要按文件名推断。',
              '- 补充材料是**已入库论文的附件**，不是独立论文条目：'
              '**不要**为它造 `_Abstract_Card.md`（会犯 K8），**不要**改语料三的条目计数。']
    (outdir / '_清单.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return outdir, names, len(body)


def corpus_pmcids(root):
    out = []
    for f in sorted(Path(root).rglob('*.xml')):
        head = f.read_text(encoding='utf-8', errors='ignore')[:20000]
        for pid in sorted(set(re.findall(r'PMC\d+', head))):
            out.append((pid, str(f)))
    return out


def main():
    ap = argparse.ArgumentParser(description='取回 Europe PMC 补充材料')
    ap.add_argument('--pmcid', action='append', default=[], help='PMCID，可多次')
    ap.add_argument('--probe', action='store_true', help='只探测不下载')
    ap.add_argument('--from-corpus', action='store_true', help='对语料三全部 XML 探测')
    ap.add_argument('--corpus-root', default='01_参考论文/生物力学线')
    ap.add_argument('--out', default=None, help='下载目录（默认 06_附录/补充材料/<PMCID>）')
    ap.add_argument('--report', default='06_附录/生物力学语料工具/epmc_supp_report.json')
    args = ap.parse_args()

    targets = list(args.pmcid)
    src = {}
    if args.from_corpus:
        for pid, f in corpus_pmcids(args.corpus_root):
            if pid not in targets:
                targets.append(pid)
            src.setdefault(pid, []).append(f)
    if not targets:
        ap.error('至少给一个 --pmcid 或 --from-corpus')

    results, got = [], []
    for pid in targets:
        r = probe(pid)
        r['source_xml'] = src.get(pid, [])
        print(f"{pid}  epmc={r.get('epmc_status')}  bytes={r.get('epmc_bytes', 0):>9}  "
              f"ncbi_oa={r.get('ncbi_oa_status')}  -> {'有补充材料' if r['has_supplementary'] else '无'}"
              + (f"  [{r['note']}]" if r.get('note') else ''))
        if r['has_supplementary'] and not args.probe:
            base = Path(args.out) if args.out else Path('06_附录/补充材料') / pid
            d = base
            d, names, nb = fetch(pid, d)
            r['saved_to'] = str(d)
            r['files'] = names
            r['n_supp'] = sum(1 for n in names if classify(n) == 'supp')
            r['n_fig'] = sum(1 for n in names if classify(n) == 'fig')
            got.append((pid, len(names), nb))
            print(f'      -> 已落盘 {d}  ({len(names)} 个文件 / {nb:,} bytes)')
        results.append(r)
        time.sleep(0.7)                                        # 礼貌限速

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({
        'tool': 'fetch_epmc_supp.py',
        'endpoint': EPMC,
        'second_channel': NCBI_OA,
        'n_targets': len(targets),
        'n_with_supplementary': sum(1 for r in results if r['has_supplementary']),
        'results': results,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    n200 = sum(1 for r in results if r['has_supplementary'])
    nreal = sum(1 for r in results if r.get('n_supp', 0) > 0)
    print(f'\n汇总：探测 {len(targets)} 个 PMCID')
    print(f'  · 接口返回 HTTP 200 + zip：{n200} 个  ← **这个数会高估，接口混装了正文插图**')
    print(f'  · ✅ **按文件名分类，真有补充材料：{nreal} 个**  ← 这个才是准数')
    print(f'  · 已落盘 {len(got)} 个 → 报告 {args.report}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
