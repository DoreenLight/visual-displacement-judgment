#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the only user-facing list: included papers still missing full text.

Research decisions live in 10_研究方向/主动参与与训练强度_链路研究/文献台账.json.
Existing corpus indexes are lookup evidence, not automatic user download tasks.
No downloaded entries, web evidence, historical statistics or research chores are
rendered here. Run from any working directory: python3 <path>/gen_acquisition_list.py.
"""
import importlib.util
from functools import lru_cache
import json
import re
import subprocess
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / '待下载论文总清单.md'
MANIFEST = ROOT / '10_研究方向/主动参与与训练强度_链路研究/文献台账.json'
BM_TOOLS = HERE / '生物力学语料工具'
EXCLUDE = {'.pi', '_translation_outputs', '_research_run', '__pycache__',
           'node_modules', 'libs', '.git', '历史归档'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def normalize_title(title):
    text = unicodedata.normalize('NFKD', title).casefold()
    return ''.join(char for char in text if char.isalnum())


def identity(paper):
    doi = (paper.get('doi') or '').strip().lower()
    doi = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', doi)
    if doi:
        return 'doi:' + doi
    arxiv = (paper.get('arxiv_id') or '').strip()
    if arxiv:
        return 'arxiv:' + re.sub(r'v\d+$', '', arxiv)
    title = normalize_title(paper.get('title') or '')
    if not title:
        raise ValueError('Paper must have a verified identity or a nonempty title')
    return 'title:' + title


def local_path(root, ref):
    if not ref:
        return None
    path = (root / ref).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        return None
    if path.suffix.lower() not in {'.pdf', '.xml'}:
        return None
    return path


def corpus_fulltexts(root=ROOT):
    """Return identity -> local fulltexts and paths already represented by indexes."""
    catalog = {}
    known = set()

    def add(paper, ref):
        path = local_path(root, ref)
        if path is not None:
            catalog.setdefault(identity(paper), []).append(path)
            known.add(path)

    for base in ['01_参考论文', '09_章隆彬_合作团队']:
        for index in (root / base).rglob('index.json'):
            if any(part in EXCLUDE or part.startswith('_backup') for part in index.parts):
                continue
            data = load(index)  # Broken declared indexes must not silently mean "no fulltext".
            for p in data.get('papers', []):
                for key in ['local_fulltext', 'fulltext_path', 'fulltext_reference',
                            'pdf_path', 'xml_path']:
                    add(p, p.get(key))
    # BM's index has no disk path; use its existing canonical status function.
    tools = root / '06_附录/生物力学语料工具'
    if tools.is_dir():
        spec = importlib.util.spec_from_file_location('biomech_for_acquisition', tools / 'build_biomech.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        resolved = load(tools / 'biomech_resolved.json')
        candidates = load(tools / 'biomech_candidates.json')
        report = load(tools / 'biomech_download_report.json')
        dl = {r['key']: r for r in report['items'] if r.get('key')}
        for p in resolved['papers']:
            if p.get('status') not in {'confirmed', 'confirmed_single'}:
                continue
            status, ref = module.disk_status(p, dl, candidates['meta']['axes']['thread'])
            if status == 'ok':
                add(p, '01_参考论文/整合系统/生物力学线/' + ref)
    return catalog, known


@lru_cache(maxsize=4096)
def pdf_head(path, size, mtime_ns, layout=False):
    # In-process cache only; the stat key prevents reuse after a user supplies a new file.
    with path.open('rb') as stream:
        if stream.read(5) != b'%PDF-':
            return ''
    args = ['pdftotext'] + (['-layout'] if layout else [])
    result = subprocess.run(args + ['-f', '1', '-l', '2', str(path), '-'],
                            capture_output=True, text=True, timeout=30, check=True)
    return result.stdout


def content_matches(path, paper):
    """Check document identity, not a DOI mentioned in a bibliography."""
    if path.suffix.lower() == '.xml':
        import xml.etree.ElementTree as ET
        tree = ET.parse(path).getroot()
        if tree.find('.//body') is None:
            return False
        ids = [e.text.strip().lower() for e in tree.findall('.//article-meta/article-id')
               if e.get('pub-id-type') == 'doi' and e.text]
        article_title = tree.find('.//article-title')
        title = ''.join(article_title.itertext()) if article_title is not None else ''
        return ((paper.get('doi') or '').lower() in ids
                or normalize_title(title) == normalize_title(paper['title']))
    stat = path.stat()
    head = pdf_head(path, stat.st_size, stat.st_mtime_ns)
    # Title is required even if a DOI appears in the first two pages.
    titles = [normalize_title(t) for t in [paper['title']] + paper.get('title_aliases', [])]
    if any(t and t in normalize_title(head) for t in titles):
        return True
    # Some older PDFs scramble the title in reading-order extraction. Retry the
    # same pages in physical layout; retain exact normalized title/verified alias
    # matching rather than accepting a DOI alone or fuzzy title similarity.
    head = pdf_head(path, stat.st_size, stat.st_mtime_ns, layout=True)
    return any(t and t in normalize_title(head) for t in titles)


def unindexed_fulltexts(root, known):
    # Also inspect indexed files: an old wrong DOI/title must not hide an existing original.
    # The caller checks indexed candidates first; PDF extraction is cached per file/stat.
    result = []
    for base in ['01_参考论文', '09_章隆彬_合作团队']:
        for path in (root / base).rglob('*'):
            if (not path.is_file() or path.suffix.lower() not in {'.pdf', '.xml'}
                    or any(part in EXCLUDE or part.startswith('_backup') for part in path.parts)):
                continue
            result.append(path.resolve())
    return result


def pending_papers(data, root=ROOT, catalog=None, unindexed=None):
    if data.get('schema') != 'rehab-chain-research/v1':
        raise ValueError('Unsupported research manifest; refusing to overwrite the list')
    if catalog is None:
        catalog, known = corpus_fulltexts(root)
        unindexed = unindexed_fulltexts(root, known)
    unindexed = unindexed or []
    merged = {}
    for p in data['papers']:
        if p.get('screening') != 'included' or p.get('needs_fulltext') is not True:
            continue
        if p.get('kind') != 'paper':
            continue
        key = identity(p)
        merged.setdefault(key, []).append(p)
    pending = []
    for key, records in merged.items():
        p = records[0]
        refs = list(catalog.get(key, []))
        refs += [path for r in records if (path := local_path(root, r.get('local_fulltext')))]
        # A duplicate index reference with different identity must still be excluded
        # by content when a newly supplied unindexed fulltext is found.
        if any(content_matches(path, p) for path in dict.fromkeys(refs + unindexed)):
            continue
        title = (p.get('title') or '').strip()
        url = (p.get('landing_url') or '').strip()
        parsed = urlparse(url)
        if not title or parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.username:
            raise ValueError('Missing verified paper title or usable landing URL: ' + key)
        pending.append({**p, 'title': title,
                        'venue': p.get('venue') or '未核实',
                        'database': p.get('database') or '未核实',
                        'landing_url': url})
    return pending


def md(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def render(papers):
    # 口径固定：只列「本轮纳入且全库复核后仍缺全文」的可取件论文，每行四项必填。
    # 不得加入优先级、历史统计或状态标记——分组与取件顺序见 06_附录/取件导读.md，
    # 那些属于导读，不属于清单（由 test_required_columns_only 强制）。
    lines = ['# 需要你下载的论文', '']
    if not papers:
        return '\n'.join(lines + ['当前无待下载论文。', ''])
    lines += ['| 序号 | 论文标题 | 地址 | 期刊/会议 | 数据库/平台 |',
              '|---|---|---|---|---|']
    for i, p in enumerate(papers, 1):
        lines.append('| %d | %s | [取件页](%s) | %s | %s |' %
                     (i, md(p['title']), p['landing_url'], md(p['venue']), md(p['database'])))
    return '\n'.join(lines + [''])


def main():
    papers = pending_papers(load(MANIFEST))
    text = render(papers)  # All checks precede write; no partial or empty list on parse failure.
    OUT.write_text(text, encoding='utf-8')
    print('生成待补全文 %d 条：%s' % (len(papers), OUT))


if __name__ == '__main__':
    main()
