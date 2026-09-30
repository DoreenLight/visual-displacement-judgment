#!/usr/bin/env bash
# ==============================================================================
# 知识库自检脚本 verify_kb.sh  (v2 — 补强 4 个盲区)
# ==============================================================================
# 用途：对 02-视觉动作分析/ 做 8 项结构与引用完整性自检。
# 运行：bash 06_附录/自检脚本/verify_kb.sh
# 退出码：0 = 全部 PASS；1 = 存在 FAIL
#
# v2 相对 v1 的补强（v1 曾报 7 PASS/0 FAIL 而同时存在 A/C/D 类真实缺口）：
#   [E1] Case 5 新增「空壳目录」检查（原只查"目录内完全无文件"，检查 0 项）
#   [E2] Case 8 新增「软链接可达性」检查（原完全不检查软链接，断链从未被发现）
#   [E3] Case 7 新增「计划覆盖率」报告，并接受 *_reconstruction.md 作为
#        PDF 的合法替代孪生（替代包是经授权的交付形态）
#   [E4] Case 6 新增「磁盘复算 vs 文档声称值」比对，数据源为文档内的
#        机器可读块 <!-- VERIFY_KB_COUNTS ... -->（原只查方块数与百分比自洽，不查分母）
#
# 设计约束：
#   - 兼容 macOS 自带 bash 3.2（不使用 associative array / mapfile 等新特性）
#   - 重逻辑放在内嵌 python3 里，避免 BSD/GNU find 差异
#   - 幂等：只读，不修改任何文件
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[FATAL] 未找到 python3，无法运行自检" >&2
  exit 1
fi

python3 - "$ROOT" <<'PYEOF'
import sys, os, re
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
EXCLUDE_PARTS = {'.pi', '_translation_outputs', 'libs', 'node_modules', '__pycache__', '.git'}

def excluded(p: Path) -> bool:
    try:
        rel = p.relative_to(ROOT)
    except ValueError:
        return True
    return any(part in EXCLUDE_PARTS for part in rel.parts)

cases = []   # (no, name, fail_count, total_count, detail_lines)

# ---------------------------------------------------------------- Case 1
# Markdown 相对链接完整性
broken, total = [], 0
for md in sorted(ROOT.rglob('*.md')):
    if excluded(md):
        continue
    txt = md.read_text(encoding='utf-8', errors='ignore')
    txt = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', txt)          # 图片单独在 Case 2 统计
    for m in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', txt):
        label, target = m.group(1), m.group(2).strip()
        if target.startswith(('http://', 'https://', '#', 'mailto:')):
            continue
        if re.match(r'^[†*✗✓\d\s.,、。·—–-]+$', label):     # 脚注/上标符号
            continue
        t = target.split('#')[0].split('?')[0]
        if not t:
            continue
        total += 1
        if not (md.parent / t).exists():
            broken.append('%s -> [%s](%s)' % (md.relative_to(ROOT), label, target))
cases.append((1, 'Markdown 相对链接完整性', len(broken), total, broken))

# ---------------------------------------------------------------- Case 2
# 翻译文档图片链接完整性（只查论文子目录内的 _全文翻译.md，不含 _translation_outputs 备份）
broken, total = [], 0
for md in sorted(ROOT.rglob('*_全文翻译.md')):
    if excluded(md):
        continue
    txt = md.read_text(encoding='utf-8', errors='ignore')
    for m in re.finditer(r'!\[[^\]]*\]\(([^)]+)\)', txt):
        src = m.group(1).strip()
        if src.startswith(('http://', 'https://', '#', 'data:')):
            continue
        total += 1
        if not (md.parent / src).exists():
            broken.append('%s -> ![]()' % md.relative_to(ROOT))
cases.append((2, '翻译文档图片链接完整性', len(broken), total, sorted(set(broken))))

# ---------------------------------------------------------------- Case 3
# 顶层目录编号唯一性
entries, dups = [], []
seen = {}
for d in sorted(ROOT.iterdir()):
    if not d.is_dir() or d.is_symlink():
        continue
    m = re.match(r'^(\d{2})[-_]', d.name)
    if m:
        entries.append(d.name)
        seen.setdefault(m.group(1), []).append(d.name)
for num, names in sorted(seen.items()):
    if len(names) > 1:
        dups.append('编号 %s 重复: %s' % (num, ' / '.join(names)))
cases.append((3, '顶层目录编号唯一性', len(dups), len(entries), dups))

# ---------------------------------------------------------------- Case 4
# 顶层目录分隔符一致性（要求 NN_主题 下划线风格；软链接别名不计入）
bad_sep, checked = [], 0
for d in sorted(ROOT.iterdir()):
    if not d.is_dir() or d.is_symlink():
        continue
    m = re.match(r'^(\d{2})([-_])(.+)$', d.name)
    if not m:
        continue
    checked += 1
    if m.group(2) == '-':
        bad_sep.append('%s 使用连字符，应改为下划线' % d.name)
cases.append((4, '顶层目录分隔符一致性', len(bad_sep), checked, bad_sep))

# ---------------------------------------------------------------- Case 5
# [E1] 空目录 + 空壳目录
#   空目录 = 目录内完全无文件
#   空壳目录 = 目录内只有一份自称"空壳/待补/待填"的 README，且承诺的文档一个都不存在
WALK_EXCLUDE = {'.pi', '.git', 'node_modules', '__pycache__'}
empty = []
scanned_dirs = 0
SHELL_MARKERS = ('当前为空壳', '待 Phase 2 填充', '待补：', '待填：', '本目录当前为空壳')
slot_bad = []
for root, dirs, files in os.walk(ROOT, followlinks=False):
    rp = Path(root)
    rel = rp.relative_to(ROOT)
    if any(part in WALK_EXCLUDE for part in rel.parts):
        dirs[:] = []
        continue
    scanned_dirs += 1
    if not dirs and not files:
        empty.append(str(rel))
    # 空壳检测：只针对"计划型"目录（README 里带承诺文档清单的）
    docfiles = [f for f in files if f.endswith('.md')]
    if 'README.md' in files and len(docfiles) == 1:
        try:
            rt = (rp / 'README.md').read_text(encoding='utf-8', errors='ignore')
        except Exception:
            rt = ''
        hit = next((mk for mk in SHELL_MARKERS if mk in rt), None)
        if hit:
            slot_bad.append('%s/ —— README 自称「%s」且无其他文档' % (rel, hit.rstrip('：')))
cases.append((5, '空目录 + 空壳目录', len(empty) + len(slot_bad), scanned_dirs,
              sorted(empty) + sorted(slot_bad)))

# ---------------------------------------------------------------- Case 6
# 索引文档一致性：4 份入口文档的「最后更新」日期必须相同
#  + 进度条图文相符（自查自）
#  + [E4] 机器可读计数块 <!-- VERIFY_KB_COUNTS ... --> 与磁盘复算值比对
DOCS = ['README.md', 'INDEX.md', 'HANDOFF.md', 'STATUS.md']
dates, missing = {}, []
for name in DOCS:
    f = ROOT / name
    if not f.exists():
        missing.append(name)
        continue
    txt = f.read_text(encoding='utf-8', errors='ignore')
    d = None
    m = re.search(r'最后更新[^\n]*?(\d{4}-\d{2}-\d{2})', txt)
    if m:
        d = m.group(1)
    else:
        m = re.search(r'^updated:\s*(\d{4}-\d{2}-\d{2})', txt, re.M)
        if m:
            d = m.group(1)
    dates[name] = d or '(未标注)'

issues = []
if missing:
    issues.append('缺失入口文档: %s' % ', '.join(missing))
uniq = set(v for v in dates.values() if v != '(未标注)')
if len(uniq) > 1:
    issues.append('「最后更新」日期不一致: ' + '; '.join('%s=%s' % kv for kv in sorted(dates.items())))
elif not uniq:
    issues.append('4 份入口文档均未标注日期')

bar_bad = []
for name in DOCS:
    f = ROOT / name
    if not f.exists():
        continue
    for ln, line in enumerate(f.read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
        for m in re.finditer(r'\[([█░]+)\]\s*(\d+)%', line):
            bars, stated = m.group(1), int(m.group(2))
            real = int(round(bars.count('█') * 100.0 / len(bars)))
            if abs(real - stated) > 1:
                bar_bad.append('%s:%d 方块=%d%% 但文字写 %d%%' % (name, ln, real, stated))
issues.extend(bar_bad)
cases.append((6, '索引一致性（日期/进度条/计数块）', len(issues), len(DOCS), issues))

# ---------------------------------------------------------------- Case 7
# [E3] PDF <-> _全文翻译.md 配对，接受 *_reconstruction.md 作为合法替代孪生
#      + 报告"论文条目覆盖率"（含替代包）
PAPER_ROOTS = [ROOT / '01_参考论文', ROOT / '09_章隆彬_合作团队']
pdf_dirs, _seen = [], set()
for d in PAPER_ROOTS:
    if d.exists() and d not in _seen:
        pdf_dirs.append(d)
        _seen.add(d)

pdfs, zhs, recs, cards = [], [], [], []
for pr in pdf_dirs:
    for f in pr.rglob('*.pdf'):
        if not excluded(f):
            pdfs.append(f)
    for f in pr.rglob('*_全文翻译.md'):
        if not excluded(f):
            zhs.append(f)
    for f in pr.rglob('*_reconstruction.md'):
        if not excluded(f):
            recs.append(f)
    for f in pr.rglob('*Abstract_Card.md'):
        if not excluded(f):
            cards.append(f)

pair_bad = []
for pdf in pdfs:
    zh = pdf.with_name(pdf.stem + '_全文翻译.md')
    if not zh.exists():
        pair_bad.append('%s 缺对应 %s' % (pdf.relative_to(ROOT), zh.name))

rec_only = []   # 只有替代包、没有 PDF 的条目（合法，但需登记）
for f in zhs:
    if not f.name.endswith('_全文翻译.md'):
        continue
    stem = f.name[:-len('_全文翻译.md')]
    pdf_twin = f.with_name(stem + '.pdf')
    rec_twin = f.with_name(stem + '_reconstruction.md')
    if pdf_twin.exists():
        continue
    if rec_twin.exists():
        rec_only.append(stem)
    else:
        pair_bad.append('%s 既无 PDF 也无 _reconstruction.md 孪生' % f.relative_to(ROOT))

total_entries = len(pdfs) + len(rec_only)
coverage = (len(pdfs) * 100.0 / total_entries) if total_entries else 0.0
cases.append((7, 'PDF <-> _全文翻译.md 配对 + 覆盖率', len(pair_bad), len(pdfs), pair_bad))

# ---------------------------------------------------------------- Case 8
# [E2] 软链接可达性（断链检查）
broken_links, all_links = [], 0
for root, dirs, files in os.walk(ROOT, followlinks=False):
    rp = Path(root)
    rel = rp.relative_to(ROOT)
    if any(part in WALK_EXCLUDE for part in rel.parts):
        dirs[:] = []
        continue
    for nm in list(dirs) + list(files):
        p = rp / nm
        if p.is_symlink():
            all_links += 1
            if not p.exists():
                broken_links.append('%s -> %s（不可达）' % (p.relative_to(ROOT), os.readlink(p)))
cases.append((8, '软链接可达性（断链检查）', len(broken_links), all_links, sorted(broken_links)))

# ---------------------------------------------------------------- Case 9
# [E5] LIST.md 陈旧状态字段：不得再声称论文「未下载/未获取/待用户/速读卡/替代包」
#       而对应 PDF 已在盘（或已全库取得）
LIST_STALE = ['未下载', '未获取', '待用户', '待实际校验',
              'IEEE 锁定', 'IEEE paywall', '⚠️ 速读卡', '⚠️ 替代包', '翻译中']
list_stale, list_checked = [], 0
for md in sorted(ROOT.rglob('LIST.md')):
    if excluded(md):
        continue
    list_checked += 1
    for ln, line in enumerate(md.read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
        if '<!-- status-ok -->' in line:
            continue
        for mk in LIST_STALE:
            if mk in line:
                list_stale.append('%s:%d 含陈旧标记「%s」→ %s' %
                                  (md.relative_to(ROOT), ln, mk, line.strip()[:70]))
                break
cases.append((9, 'LIST.md 陈旧状态字段', len(list_stale), list_checked, sorted(list_stale)))

# ---------------------------------------------------------------- Case 10
# [E5] 正文计数声明 vs 磁盘复算：消除「计数块正确但正文写错」的双口径问题
def _in_quotes(line, pos):
    """判定该字符位置是否落在「…」或 `…` 内（引用历史值时豁免）。"""
    # 「…」
    i = 0
    while True:
        a = line.find('「', i)
        if a < 0:
            break
        b = line.find('」', a + 1)
        if b < 0:
            break
        if a < pos < b:
            return True
        i = b + 1
    # `…`：按反引号分段，奇数段为代码/引用
    parts = line.split('`')
    acc = 0
    for k, seg in enumerate(parts):
        if k % 2 == 1 and acc <= pos < acc + len(seg):
            return True
        acc += len(seg) + 1
    return False


CLAIM_PATTERNS = [
    (re.compile(r'(\d+)\s*篇\s*PDF'), 'pdf', len(pdfs), 'PDF'),
    (re.compile(r'(?<![\d,])PDF\s*(\d+)(?!\s*[,\d])'), 'pdf', len(pdfs), 'PDF(后置)'),
    (re.compile(r'(\d+)\s*篇\s*(?:_zh\.md|中文翻译)'), 'zh', len(zhs), '中文翻译'),
    (re.compile(r'(\d+)\s*篇论文'), 'entries', total_entries, '论文'),
]
count_claims = []
claims_checked = 0
# 只检查「全库总数」声明，排除子目录计数 / 代码树 / 历史引述 / 行动项
SUBDIR_HINTS = ('01_参考论文', '09_章隆彬', 'sEMG_原始参照论文', 'RGB_', 'ViT_HMR', 'TokenHMR',
                'PoseFormer', 'METRO', 'OpenCap', '腕关节康复感知', 'papers/', '代表性', '该目录')
for name in DOCS:
    f = ROOT / name
    if not f.exists():
        continue
    in_fence = False
    for ln, line in enumerate(f.read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
        if line.strip().startswith('```'):
            in_fence = not in_fence
            continue
        if in_fence:
            continue                                  # 代码块/目录树内的计数为子目录级
        if any(h in line for h in SUBDIR_HINTS):
            continue
        if any(k in line for k in ('虚报', '补全前', '变更记录', '<!-- count-ok -->')):
            continue
        if re.search(r'(翻译|跑|batch)\s*\d+\s*篇', line):
            continue                                  # 行动项而非计数声明
        for rx, key, expect, label in CLAIM_PATTERNS:
            for m in rx.finditer(line):
                if _in_quotes(line, m.start()):
                    continue                          # 该匹配本身在引号/反引号内 → 历史值
                got = int(m.group(1))
                claims_checked += 1
                if got != expect:
                    count_claims.append('%s:%d 声称「%s」但磁盘实际 %d → %s' %
                                        (name, ln, m.group(0), expect, line.strip()[:70]))
cases.append((10, '正文计数声明 vs 磁盘', len(count_claims), claims_checked, sorted(count_claims)))

# ---------------------------------------------------------------- Case 11
# [E5] 目录树/清单中的「子目录 ← N 篇 PDF」与技术无关但同样会漂移
def _pdf_count_of(dirname):
    nm = dirname.rstrip('/')
    if not nm:
        return None
    for d in ROOT.rglob(nm):
        if d.is_dir() and not excluded(d):
            return len([f for f in d.rglob('*.pdf') if not excluded(f)])
    return None

SUBDIR_CLAIM_RE = re.compile(r'([^\s├└│─]{2,})/\s*[←★\s]+?(?:(\d+)\s*篇\s*PDF|(\d+)\s*PDF)')
tree_claims = []
tree_checked = 0
for name in DOCS:
    f = ROOT / name
    if not f.exists():
        continue
    for ln, line in enumerate(f.read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
        if '<!-- count-ok -->' in line:
            continue
        for m in SUBDIR_CLAIM_RE.finditer(line):
            if _in_quotes(line, m.start()):
                continue
            dname = m.group(1)
            got = int(m.group(2) or m.group(3))
            real = _pdf_count_of(dname)
            tree_checked += 1
            if real is not None and real != got:
                tree_claims.append('%s:%d 「%s/」声称 %d 篇 PDF，实际 %d' %
                                   (name, ln, dname, got, real))
cases.append((11, '目录树中的子目录 PDF 计数', len(tree_claims), tree_checked, sorted(tree_claims)))

# ---------------------------------------------------------------- Case 12
# [E5] 正文声明的 PDF 字节数 vs 磁盘实际（字节数同样会漂移）
# 只接受「文件名 紧邻 字节数」的写法：中间片段不得含中文或竖线，避免把同行的「速读卡 5,578 B」误绑到 PDF 文件名上
SIZE_CLAIM_RE = re.compile(r'([A-Za-z0-9_\-\.]+\.pdf)[\s`\u2018\u2019（）()、,，:：*\-–—]{0,12}?([\d]{1,3}(?:,\d{3})+)\s*(?:B|字节)')
def _find_pdf(basename):
    for f in ROOT.rglob(basename):
        if f.is_file() and not excluded(f):
            return f
    return None
size_claims, size_checked = [], 0
for md in sorted(ROOT.rglob('*.md')):
    if excluded(md):
        continue
    for ln, line in enumerate(md.read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
        if '<!-- count-ok -->' in line:
            continue
        for m in SIZE_CLAIM_RE.finditer(line):
            base, claimed = m.group(1), int(m.group(2).replace(',', ''))
            f = _find_pdf(base)
            if f is None:
                continue
            size_checked += 1
            real = f.stat().st_size
            if real != claimed:
                size_claims.append('%s:%d 「%s」声称 %s B，实际 %s B' %
                                   (md.relative_to(ROOT), ln, base, format(claimed, ','), format(real, ',')))
cases.append((12, '正文 PDF 字节数 vs 磁盘', len(size_claims), size_checked, sorted(size_claims)))

# ---------------------------------------------------------------- Case 13
# [E9] 速读卡 / 中文精读里不得再声称「原文未获取」（*_reconstruction.md 已归档，豁免）
#       这类陈述与磁盘上已存在的 PDF 直接矛盾
CARD_STALE = ['暂未获取', 'PDF 未下载', '全文：**未获取**', '暂无 PDF',
              '机构订阅后', '删掉本卡', '一旦用户放入', 'IEEE 锁定', 'IEEE paywall',
              'Elsevier 锁定', 'Springer 锁定', '锁定，', '速读卡见']
card_stale, card_checked = [], 0
for pat in ('*_Abstract_Card.md', '*_全文翻译.md'):
    for md in sorted(ROOT.rglob(pat)):
        if excluded(md):
            continue
        card_checked += 1
        for ln, line in enumerate(md.read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
            if '<!-- status-ok -->' in line:
                continue
            for mk in CARD_STALE:
                if mk in line:
                    card_stale.append('%s:%d 含「%s」→ %s' %
                                      (md.relative_to(ROOT), ln, mk, line.strip()[:60]))
                    break
cases.append((13, '速读卡/精读中的「未获取」陈述', len(card_stale), card_checked, sorted(card_stale)))

# ---------------------------------------------------------------- Case 14
# [E10] 归档的 *_reconstruction.md 不得再以「替代包/未获原文」自述
ARCHIVE_STALE = ['深度替代包（来源受限）', '未获原文，细节为重建', '原文到位后必须逐条覆盖',
                 '证据等级**：`替代包（来源受限）`']
arch_stale, arch_checked = [], 0
for md in sorted(ROOT.rglob('*_reconstruction.md')):
    if excluded(md):
        continue
    arch_checked += 1
    for ln, line in enumerate(md.read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
        if '<!-- status-ok -->' in line:
            continue
        for mk in ARCHIVE_STALE:
            if mk in line:
                arch_stale.append('%s:%d 含「%s」' % (md.relative_to(ROOT), ln, mk))
                break
cases.append((14, '归档包不得自述「替代包/未获原文」', len(arch_stale), arch_checked, sorted(arch_stale)))

# ---------------------------------------------------------------- Case 15
# [E10] 文档中的「N 项缺口」「N 用例」自述数与实际一致
count_consistency = []
cons_checked = 0
# (a) 台账合计 vs 各处「N 项缺口」
ledger_f = ROOT / '06_附录' / '缺口台账.md'
ledger_total = None
if ledger_f.exists():
    m = re.search(r'\|\s*\*\*合计\*\*\s*\|\s*\*\*(\d+)\*\*', ledger_f.read_text(encoding='utf-8'))
    if m:
        ledger_total = int(m.group(1))
scan_files = list(DOCS) + ['06_附录/缺口台账.md']
if ledger_total is not None:
    for name in scan_files:
        f = ROOT / name
        if not f.exists():
            continue
        for ln, line in enumerate(f.read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
            if any(k in line for k in ('建档', '<!-- count-ok -->')):
                continue
            for mm in re.finditer(r'(\d+)\s*\*{0,2}\s*项\s*\*{0,2}\s*缺口', line):
                cons_checked += 1
                v = int(mm.group(1))
                if v != ledger_total:
                    count_consistency.append('%s:%d 称「%s 项缺口」但台账合计为 %d' %
                                             (name, ln, v, ledger_total))
# (b) 脚本实际用例数 vs 文档声称
script_f = ROOT / '06_附录' / '自检脚本' / 'verify_kb.sh'
if script_f.exists():
    # 统计「不同用例编号」的个数（避免把本行的字面量也数进去）
    actual_n = len(set(re.findall(r"cases\.append\(\((\d+(?:\.\d+)?)\s*,", script_f.read_text(encoding='utf-8'))))
    for name in scan_files:
        f = ROOT / name
        if not f.exists():
            continue
        for ln, line in enumerate(f.read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
            if any(k in line for k in ('新增', '升 v', '补 E', '<!-- count-ok -->')):
                continue
            for mm in re.finditer(r'(\d+)\s*个?用例', line):
                cons_checked += 1
                v = int(mm.group(1))
                if v != actual_n:
                    count_consistency.append('%s:%d 称「%s」但脚本实为 %d 个用例' %
                                             (name, ln, mm.group(0).strip(), actual_n))
cases.append((15, '缺口数/用例数自述一致性', len(count_consistency), cons_checked, sorted(count_consistency)))

# ---------------------------------------------------------------- Case 16
# [E11] 「待查/待校验/待用户/待补/待填」类**无结论占位**不得出现在「现役」正文里
#       历史记录（发现时描述、变更记录）与归档包豁免
PLACEHOLDER_TOKENS = ['待校验', '待查', '待用户', '待手动', '待补', '待填']
FINDING_ROW_RE = re.compile(r'^\|\s*\*{0,2}[ABCDE]\d*\*{0,2}\s*\|')   # 台账发现表的行
HIST_HINTS = ('变更记录', '发现时', '历史', '归档说明', '不得再', '规则', '拦截',
              '检索渠道尝试记录', '<!-- ', 'Falsification', '反证')
ph_hits, ph_checked = [], 0
for md in sorted(ROOT.rglob('*.md')):
    if excluded(md):
        continue
    if md.name.endswith('_reconstruction.md'):
        continue                                   # 归档包已在 Case 14 覆盖
    rel = str(md.relative_to(ROOT))
    if rel.startswith('06_附录/检索渠道尝试记录'):
        continue
    in_fence = False
    for ln, line in enumerate(md.read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
        if line.strip().startswith('```'):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if FINDING_ROW_RE.match(line.strip()):
            continue                               # 台账发现表（历史描述）
        if any(h in line for h in HIST_HINTS):
            continue
        ph_checked += 1
        for tk in PLACEHOLDER_TOKENS:
            if tk in line:
                ph_hits.append('%s:%d 含无结论占位「%s」→ %s' %
                               (rel, ln, tk, line.strip()[:64]))
                break
cases.append((16, '现役正文中的无结论占位', len(ph_hits), ph_checked, sorted(ph_hits)))

# ---------------------------------------------------------------- Case 17
# [E12] 不得把已归档的 *_reconstruction.md 当作**权威来源**
AUTH_STALE = ['以重建包为准', '由重建包更正', '重建包为准', '见重建包为准']
auth_hits, auth_checked = [], 0
for md in sorted(ROOT.rglob('*.md')):
    if excluded(md):
        continue
    if md.name.endswith('_reconstruction.md'):
        continue
    rel = str(md.relative_to(ROOT))
    if rel.startswith('06_附录/检索渠道尝试记录'):
        continue
    in_fence = False
    for ln, line in enumerate(md.read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
        if line.strip().startswith('```'):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if any(h in line for h in ('变更记录', '发现时', '历史', '<!-- ', '反证', '拦截', '规则')):
            continue
        if FINDING_ROW_RE.match(line.strip()):
            continue                               # 台账发现表（含 E12 自述行）
        auth_checked += 1
        for tk in AUTH_STALE:
            if tk in line:
                auth_hits.append('%s:%d 把归档包当权威「%s」→ %s' % (rel, ln, tk, line.strip()[:60]))
                break
cases.append((17, '不得把归档包当权威来源', len(auth_hits), auth_checked, sorted(auth_hits)))

# ---------------------------------------------------------------- [E4] 计数块比对
# 必须要求标签后直接换行，避免匹配到正文里对“<!-- VERIFY_KB_COUNTS -->”的引用
COUNT_RE = re.compile(r'<!--\s*VERIFY_KB_COUNTS\s*\n(.*?)-->', re.S)
declared, declared_src = {}, None
for name in DOCS:
    f = ROOT / name
    if not f.exists():
        continue
    m = COUNT_RE.search(f.read_text(encoding='utf-8', errors='ignore'))
    if not m:
        continue
    cand = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            cand[k.strip()] = v.strip()
    if cand:                      # 必须是真正含 key=value 的块
        declared, declared_src = cand, name
        break

actual = {
    'top_dirs':      len([d for d in ROOT.iterdir() if d.is_dir() and not d.name.startswith('.')]),
    'pdf':           len(pdfs),
    'zh':            len(zhs),
    'reconstruction': len(recs),
    'rec_only':      len(rec_only),
    'entries':       total_entries,
    'abstract_cards': len(cards),
    'broken_links':  len(broken_links),
}

count_issues = []
if not declared:
    count_issues.append('未在任何入口文档中找到 <!-- VERIFY_KB_COUNTS ... --> 机器可读计数块')
else:
    for k, v in actual.items():
        if k not in declared:
            count_issues.append('计数块缺字段 %s（磁盘实际 %s）' % (k, v))
            continue
        try:
            dv = int(declared[k])
        except ValueError:
            count_issues.append('计数块 %s=%s 非整数' % (k, declared[k]))
            continue
        if dv != v:
            count_issues.append('计数块 %s=%d 与磁盘实际 %d 不符' % (k, dv, v))
cases.append((6.5, '索引计数块 vs 磁盘复算', len(count_issues), len(actual), count_issues))

# ---------------------------------------------------------------- 输出报告
LINE = '=' * 78
print(LINE)
print('知识库自检报告 (v2)  root=%s' % ROOT)
print(LINE)

failed = 0
for no, name, fails, tots, details in cases:
    status = 'PASS' if fails == 0 else 'FAIL'
    if fails:
        failed += 1
    print('[%s] Case %-4s %-30s  失败 %-5s / 检查 %s' % (status, no, name, fails, tots))
    for d in details[:12]:
        print('         - %s' % d)
    if len(details) > 12:
        print('         ... 另有 %d 条' % (len(details) - 12))

print(LINE)
print('附加统计（磁盘复算，供参考）')
print('  顶层目录数        : %d' % actual['top_dirs'])
print('  论文 PDF 数       : %d' % actual['pdf'])
print('  翻译 _全文翻译.md 数 : %d' % actual['zh'])
print('  深度替代包        : %d  （%s）' % (actual['reconstruction'],
      ', '.join(sorted(rec_only)) if rec_only else '无'))
print('  论文条目总数      : %d  = PDF %d + 仅替代包 %d' % (total_entries, actual['pdf'], actual['rec_only']))
print('  PDF 覆盖率        : %.1f%%  （内容覆盖率 = 100%%）' % coverage)
print('  速读卡            : %d' % actual['abstract_cards'])
print('  软链接 / 断链     : %d / %d' % (all_links, actual['broken_links']))
if declared_src:
    print('  计数块来源        : %s' % declared_src)
print(LINE)
print('汇总：%d 个用例，PASS %d / FAIL %d' % (len(cases), len(cases) - failed, failed))
print(LINE)

sys.exit(1 if failed else 0)
PYEOF

exit $?
