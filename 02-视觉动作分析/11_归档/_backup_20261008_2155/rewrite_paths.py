#!/usr/bin/env python3
"""三分法重组后的路径重写器（一次性迁移工具，迁移完成后即可退役）。

做两件事，且只做这两件：
  1) 组内逃逸相对链接 `../../` -> `../../../`（子目录下沉一层）
  2) 项目根相对字面量 `01_参考论文/<X>` -> `01_参考论文/<组>/<X>`
     以及 python 式 `ROOT / '01_参考论文' / '<X>'` -> 插入组名

不改写：05_进展日志/、11_归档/、_translation_outputs/、.git/、.pi/
不改写：*_report.json / *核查报告.json 等一次性报告工件（留存当时事实）
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {}
for line in (Path(__file__).parent / '移动前清单' / '移动映射.tsv').read_text('utf-8').splitlines():
    if line.strip():
        child, group = line.split()
        GROUPS[child] = group

SKIP_DIRS = {'.git', '.pi', '05_进展日志', '11_归档', '_translation_outputs', 'images'}
SKIP_NAME_RE = re.compile(r'(_report\.json$|核查报告\.json$|\.bak[^/]*$|\.log$|\.txt$|^\.DS_Store$)')


def editable(p: Path) -> bool:
    rel = p.relative_to(ROOT)
    if any(part in SKIP_DIRS or part.startswith('_backup') for part in rel.parts):
        return False
    return not SKIP_NAME_RE.search(p.name)


def rewrite(text: str, inside_moved: bool) -> tuple[str, int, int]:
    n_esc = n_root = 0
    if inside_moved:
        text, n_esc = re.subn(r'(?<![\w/])\.\./\.\./(?=\d{2}_)', lambda m: '../../../', text)
    for child, group in GROUPS.items():
        text, k = re.subn(r'(01_参考论文/)' + re.escape(child) + r'(?![A-Za-z0-9_一-鿿])',
                           r'\g<1>' + group + '/' + child, text)
        n_root += k
        text, k = re.subn(r"((?:ROOT|Root|root|BASE|BASE_DIR|HERE|Path\(__file__\)\.resolve\(\)\.parents\[\d\])\s*/\s*['\"]01_参考论文['\"]\s*/\s*['\"])"
                           + re.escape(child) + r"['\"]", r"\g<1>" + group + "/" + child + "'\"", text)
        n_root += k
    return text, n_esc, n_root


changed, esc_total, root_total = [], 0, 0
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or p.is_symlink() or not editable(p):
        continue
    if p.suffix.lower() not in {'.md', '.json', '.py', '.sh', '.html', '.txt', '.yaml', '.csv'}:
        continue
    try:
        original = p.read_text(encoding='utf-8')
    except (UnicodeDecodeError, ValueError):
        continue
    rel = p.relative_to(ROOT)
    inside = rel.parts[:2] == ('01_参考论文', next((g for g in ('运动理解', '康复辅助', '整合系统')
                                                  if rel.parts[1:2] == (g,)), ''))
    updated, n_esc, n_root = rewrite(original, inside)
    if updated != original:
        p.write_text(updated, encoding='utf-8')
        changed.append((str(rel), n_esc, n_root))
        esc_total += n_esc
        root_total += n_root

print(json.dumps({
    'group_map': GROUPS,
    'files_changed': len(changed),
    'escaping_links_fixed': esc_total,
    'root_literals_fixed': root_total,
    'detail': [{'file': f, 'escaping': e, 'root': r} for f, e, r in changed],
}, ensure_ascii=False, indent=1))
sys.exit(0)
