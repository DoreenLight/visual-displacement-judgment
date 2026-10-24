#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mine_fulltext.py — 从已取得的全文工件中抽取**驱动设计决策的具体事实**

为什么需要：
    语料三是「摘要级定位卡」，能支撑判断相关性，但**不能支撑写论文**。
    真正能落进方法设计的是**带出处的具体事实**：坐标系怎么定义、误差量级多少、
    某参数化有没有某个自由度。Europe PMC JATS XML 提供了带 `<body>` 的完整正文，
    正适合做这件事 —— 且比 PDF 更好定位（能取到章节标题）。

纪律：
    · 每条抽取都带 `来源文件` + `章节` + 原文片段，**不改写、不翻译、不推断**
    · 只从 `<body>` 内抽取（排除 front-matter 元数据区）
    · 片段是**原文**，引用进论文前仍须回原文核对（本工具不替代核对）

用法：python3 mine_fulltext.py --pattern "ISB" --window 320
"""
import argparse
import glob
import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = (HERE / '../../01_参考论文/生物力学线').resolve()


def strip_tags(s):
    s = re.sub(r'<xref[^>]*>.*?</xref>', ' ', s, flags=re.S)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = html.unescape(s)
    return ' '.join(s.split())


def body_of(xml_text):
    m = re.search(r'<body[^>]*>(.*)</body>', xml_text, re.S)
    return m.group(1) if m else ''


def sections_with_text(body):
    """按 <sec> 递归取出 (章节标题, 该节纯文本)，保留可定位的层级。"""
    out = []

    def walk(node, prefix):
        for m in re.finditer(r'<sec\b[^>]*>(.*?)</sec>', node, re.S):
            inner = m.group(1)
            t = re.search(r'<title[^>]*>(.*?)</title>', inner, re.S)
            title = strip_tags(t.group(1)) if t else ''
            full = (prefix + ' › ' + title).strip(' ›') if title else prefix
            # 本节直接内容（去掉嵌套 sec 的部分）
            direct = re.sub(r'<sec\b.*?</sec>', ' ', inner, flags=re.S)
            txt = strip_tags(direct)
            if txt:
                out.append((full or '(无标题节)', txt))
            walk(inner, full)
    walk(body, '')
    return out


PRESETS = {
    'isb': ('ISB 坐标系约定', r'ISB|International Society of Biomechanics|joint coordinate system'),
    'wrist': ('腕/前臂 DoF 与定义', r'\bwrist\b|pronation|supination|radial|ulnar deviation'),
    'dof': ('参数化与自由度', r'SMPL-?H|\bMANO\b|kinematic (tree|chain)|degrees of freedom|\bDoF\b'),
    'error': ('误差量级与协议', r'root mean square|\bRMSE\b|mean absolute error|\bICC\b|intraclass|standard error'),
    # ---- 2026-10-01 新增：面向 V6a / V6c / V7 的定向预设 ----
    'chain': ('串联链拓扑与关节切分',
              r'kinematic (tree|chain|model|structure)|serial chain|bone hierarchy|'
              r'parent (bone|joint|node)|degrees? of freedom|\bDoFs?\b'),
    'd435i': ('深度相机 / 单目 RGB-D 上肢精度',
             r'D435|RealSense|RGB-?D|depth camera|monocular|occlusion|'
             r'self-occlusion|viewpoint|joint angle error'),
}


# ---------------------------------------------------------------- PDF 通路
# 为什么加：
#   语料三的**全文主体是 30 份 PDF，而早期版本只挖 11 份 JATS XML** ——
#   证据面窄了将近三倍，而 ISB 坐标约定、DoF 表、误差量级恰恰大量出现在 PDF 里。
#
# 做法：
#   调 `pdftotext`（poppler，本机已装）转纯文本，再按**启发式标题行**切段。
#   PDF 没有 XML 那样的 `<title>`，所以用编号标题（`2.1 Kinematics`、`III. Methods`）
#   与全大写短行来判定标题；判不出就退化为按页切，**并在输出里标明退化了**。
#
# 纪律与 XML 通路一致：原文照录、标注来源、不改写不翻译。
PDFTOTEXT = shutil.which('pdftotext')

# 形如 "2.1 Kinematics" / "2 Methods" / "III. METHODS" / "IV. RESULTS"
_HEAD_NUM = re.compile(r'^\s*(\d+(?:\.\d+)*\.?|[IVXLC]+\.)\s+(\S.{0,70})$')
# 纯大写短行（排除含句末标点的正文行）
_HEAD_CAPS = re.compile(r'^\s*([A-Z][A-Z0-9 \-&:,/\(\)]{3,60})\s*$')


def pdf_to_sections(path):
    """PDF → [(章节标题, 该段纯文本)]；判不出标题时用 (p.N) 退化标记。"""
    if not PDFTOTEXT:
        return []
    try:
        raw = subprocess.run([PDFTOTEXT, '-q', str(path), '-'],
                             capture_output=True, timeout=120).stdout.decode('utf-8', 'ignore')
    except Exception:
        return []
    if not raw.strip():
        return []
    # pdftotext 用 \f 分页；先按页留痕，便于退化时定位
    pages = raw.split('\f')
    out, cur_title, buf = [], None, []
    seen_heading = False

    def flush():
        if buf and ''.join(buf).strip():
            out.append((cur_title or '(未识别标题)', ' '.join(' '.join(buf).split())))

    for pno, page in enumerate(pages, 1):
        for line in page.split('\n'):
            s = line.strip()
            if not s:
                continue
            m = _HEAD_NUM.match(s) or _HEAD_CAPS.match(s)
            is_head = bool(m) and len(s) <= 78
            if is_head:
                flush()
                buf = []
                cur_title = s
                seen_heading = True
            else:
                buf.append(s)
    flush()
    if seen_heading:
        return out
    # 退化：完全没识别出标题 → 按页切，标题里标出页码
    return [('(p.%d · 未识别标题)' % i, ' '.join(p.split()))
            for i, p in enumerate(pages, 1) if p.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', default='')
    ap.add_argument('--preset', default='', choices=[''] + list(PRESETS))
    ap.add_argument('--window', type=int, default=300)
    ap.add_argument('--per-file', type=int, default=4)
    ap.add_argument('--out', default='')
    ap.add_argument('--source', default='both', choices=['both', 'xml', 'pdf'],
                    help='挖哪些全文源。both=XML+PDF（PDF 是全文主体，2026-10-01 起默认纳入）')
    args = ap.parse_args()

    if args.preset:
        label, pat = PRESETS[args.preset]
        rx = re.compile(pat, re.I)
    else:
        label, rx = ('自定义「%s」' % args.pattern, re.compile(args.pattern, re.I))

    blocks = []
    n_files = {'xml': 0, 'pdf': 0}

    if args.source in ('both', 'xml'):
        for f in sorted(glob.glob(str(CORPUS / '*' / '*.xml'))):
            raw = Path(f).read_text(encoding='utf-8', errors='ignore')
            body = body_of(raw)
            if not body:
                continue
            n_files['xml'] += 1
            rel = Path(f).relative_to(CORPUS)
            got = 0
            for sec, txt in sections_with_text(body):
                if got >= args.per_file:
                    break
                for m in rx.finditer(txt):
                    a = max(0, m.start() - args.window // 2)
                    b = min(len(txt), m.end() + args.window)
                    blocks.append((str(rel), sec, txt[a:b]))
                    got += 1
                    break                       # 每节最多取一处，避免灌水
                if got >= args.per_file:
                    break

    if args.source in ('both', 'pdf'):
        if not PDFTOTEXT:
            print('! 未找到 pdftotext，跳过 PDF 通路', file=sys.stderr)
        for f in sorted(glob.glob(str(CORPUS / '*' / '*.pdf'))):
            secs = pdf_to_sections(Path(f))
            if not secs:
                continue
            n_files['pdf'] += 1
            rel = Path(f).relative_to(CORPUS)
            got = 0
            for sec, txt in secs:
                if got >= args.per_file:
                    break
                for m in rx.finditer(txt):
                    a = max(0, m.start() - args.window // 2)
                    b = min(len(txt), m.end() + args.window)
                    blocks.append((str(rel), sec, txt[a:b]))
                    got += 1
                    break
                if got >= args.per_file:
                    break

    # 稳定排序：按文件名，避免 PDF 排在 XML 前导致输出抖动
    blocks.sort(key=lambda x: (x[0], x[1]))

    print('=' * 104)
    print('主题：%s    命中 %d 段' % (label, len(blocks)))
    print('来源：JATS XML %d 篇 + PDF %d 篇（源=%s）' % (n_files['xml'], n_files['pdf'], args.source))
    print('=' * 104)
    out = ['# 全文事实抽取：%s' % label, '',
           '> 由 `mine_fulltext.py` 从语料三已取得的全文中抽取：'
           'Europe PMC JATS XML 的 `<body>`（结构化章节）+ 官方 OA PDF（`pdftotext`，按启发式标题切段）。',
           '> **片段是原文，未改写、未翻译。引用进论文前仍须回原文核对。**',
           '> PDF 无结构化章节标记，`(p.N · 未识别标题)` 表示该 PDF 未能识别标题、按页退化切分。', '']
    for rel, sec, frag in blocks:
        print('\n▸ %s  ·  %s' % (rel, sec))
        print('  …%s…' % frag)
        out += ['### %s ｜ %s' % (rel, sec), '', '> %s' % frag, '']
    if args.out:
        Path(args.out).write_text('\n'.join(out), encoding='utf-8')
        print('\n已写出：%s' % args.out)


if __name__ == '__main__':
    main()
