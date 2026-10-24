#!/usr/bin/env python3
"""本次对账的只读复算记录；复用现行清单生成器，不另建下载规则。

从任意工作目录执行本文件，JSON写到stdout；不修改台账、原件或清单。
需要Python 3.9+、pdftotext和pdfinfo。结果对应执行时磁盘，不代替方法精读。
"""
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location('acquisition_audit', ROOT / '06_附录/gen_acquisition_list.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=60).stdout


def compact(text, limit=1500):
    return re.sub(r'\s+', ' ', text).strip()[:limit]


def check(condition, message):
    if not condition:
        raise AssertionError(message)


snapshots = {}
for filename in ['整理前文本快照.json', '直接索引修改前快照.json', '证据卡修改前快照.json']:
    for name, record in json.loads((HERE / filename).read_text()).items():
        check(hashlib.sha256(record['text'].encode()).hexdigest() == record['sha256'], '快照校验失败: ' + name)
        check(name not in snapshots or snapshots[name] == record, '快照冲突: ' + name)
        snapshots[name] = record
ledger_rel = str(A.MANIFEST.relative_to(ROOT))
before = json.loads(snapshots[ledger_rel]['text'])
data = A.load(A.MANIFEST)
preview = {p['id']: p for p in A.load(HERE / '台账修改预演.json')['records']}
check(len(data['papers']) == 87, '纳入数不是87')
check(len({A.identity(p) for p in data['papers']}) == 87, '身份重复')
check([(p['id'], p['title'], p.get('doi'), p.get('arxiv_id')) for p in before['papers']] ==
      [(p['id'], p['title'], p.get('doi'), p.get('arxiv_id')) for p in data['papers']], '身份/次序改变')
check(data['checked_at'] == before['checked_at'] and data['cutoff'] == before['cutoff'], '历史检索时间改变')
allowed_fields = {'local_fulltext', 'needs_fulltext', 'title_aliases', 'alias_verified_from'}
records, changes, field_counts, method_counts = [], [], Counter(), Counter()
for old, paper in zip(before['papers'], data['papers']):
    pid = paper['id']
    changed = {k: {'old': old.get(k), 'new': paper.get(k)} for k in sorted(old.keys() | paper.keys())
               if old.get(k) != paper.get(k)}
    check(set(changed) <= allowed_fields, '非授权字段改变: ' + pid)
    if changed:
        changes.append({'id': pid, 'changes': changed})
        field_counts.update(changed.keys())
    check(paper['screening'] == 'included' and paper['kind'] == 'paper' and paper['needs_fulltext'] is True,
          '记录未参加全文核验: ' + pid)
    path = A.local_path(ROOT, paper['local_fulltext'])
    check(path is not None, '原件不存在/越界/格式不合规: ' + pid)
    digest = sha(path)
    check(digest == preview[pid]['sha256'], '与写入前原件字节不一致: ' + pid)
    check(not paper.get('sha256') or digest == paper['sha256'], '与台账既有SHA不一致: ' + pid)
    check(A.content_matches(path, paper), '正文身份不匹配: ' + pid)
    rec = {'id': pid, 'identity': A.identity(paper), 'title': paper['title'],
           'doi': paper.get('doi'), 'arxiv_id': paper.get('arxiv_id'),
           'local_fulltext': paper['local_fulltext'], 'resolved_path': str(path.relative_to(ROOT)),
           'format': path.suffix[1:], 'bytes': path.stat().st_size, 'sha256': digest,
           'matches_before_sha256': True, 'declared_sha256_present': bool(paper.get('sha256')),
           'declared_version': paper.get('version'), 'declared_version_note': paper.get('version_note'),
           'identity_pass': True, 'structure_scope': '文件可解析、正文结构与末页线索；非逐页人工完整性审计'}
    if path.suffix.lower() == '.pdf':
        info = command(['pdfinfo', str(path)])
        pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M).group(1))
        check(pages > 1, 'PDF仅一页，须人工复核: ' + pid)
        full = command(['pdftotext', str(path), '-'])
        page_texts = full.split('\f')
        if not page_texts[-1].strip():
            page_texts.pop()
        check(len(page_texts) == pages, 'PDF提取页数不一致: ' + pid)
        check(len(full.strip()) > 1000, '正文过短，须人工复核: ' + pid)
        titles = [('canonical', paper['title'])] + [('verified_alias', t) for t in paper.get('title_aliases', [])]
        stat = path.stat()
        match = None
        for layout in (False, True):
            head = A.pdf_head(path, stat.st_size, stat.st_mtime_ns, layout=layout)
            for title_kind, title in titles:
                if A.normalize_title(title) in A.normalize_title(head):
                    match = {'method': ('pdf_layout_' if layout else 'pdf_reading_order_') + title_kind,
                             'matched_title': title, 'head_excerpt': compact(head)}
                    break
            if match:
                break
        check(match is not None, '匹配方法不可重现: ' + pid)
        rec.update(match)
        rec.update({'pages': pages, 'extracted_pages': len(page_texts),
                    'text_characters': len(full),
                    'empty_text_pages': [i + 1 for i, t in enumerate(page_texts) if not t.strip()],
                    'reference_heading_found': bool(re.search(r'\b(?:references|bibliography)\b', full, re.I)),
                    'last_page_excerpt': compact(page_texts[-1]), 'pdfinfo': info.strip()})
    else:
        tree = ET.parse(path).getroot()
        body = tree.find('.//body')
        body_text = ''.join(body.itertext()) if body is not None else ''
        title = tree.find('.//article-title')
        rec.update({'method': 'xml_article_metadata_and_body',
                    'article_title': ''.join(title.itertext()) if title is not None else '',
                    'article_dois': [e.text for e in tree.findall('.//article-meta/article-id')
                                     if e.get('pub-id-type') == 'doi'],
                    'body_characters': len(body_text), 'body_sections': len(body.findall('.//sec')),
                    'references': len(tree.findall('.//ref-list/ref')),
                    'body_head_excerpt': compact(body_text), 'body_tail_excerpt': compact(body_text[-1500:])})
        check(len(body_text) > 1000, 'XML正文过短: ' + pid)
    check(sha(path) == digest, '核验期间原件发生变化: ' + pid)
    method_counts[rec['method']] += 1
    records.append(rec)

pending = A.pending_papers(data)
check(not pending, '发现实际缺件')
check(A.OUT.read_text() == A.render(pending), '唯一清单不符合生成器输出')
index_updates = []
for name in ['01_参考论文/整合系统/决策模型线_JEV/index.json', '01_参考论文/康复辅助/康复整合链路/index.json']:
    old, new = json.loads(snapshots[name]['text']), A.load(ROOT / name)
    check(len(old['papers']) == len(new['papers']), '索引成员数改变: ' + name)
    for o, n in zip(old['papers'], new['papers']):
        fields = {k: {'old': o.get(k), 'new': n.get(k)} for k in o.keys() | n.keys() if o.get(k) != n.get(k)}
        check(set(fields) <= {'local_fulltext', 'pdf_path', 'fulltext_path'}, '索引非路径字段改变: ' + name)
        if fields:
            check(A.identity(n) in {A.identity(p) for p in data['papers']}, '改动超出87条: ' + n['title'])
            for change in fields.values():
                check(A.local_path(ROOT, change['new']) is not None, '索引新路径无效')
            index_updates.append({'index': name, 'title': n['title'], 'changes': fields})

chain_checks = []
for paper in A.load(ROOT / '01_参考论文/康复辅助/康复整合链路/index.json')['papers']:
    path = A.local_path(ROOT, paper['local_fulltext'])
    check(path is not None and sha(path) == paper['sha256'], '康复子库原件/SHA异常: ' + paper['title'])
    rec = {'title': paper['title'], 'path': paper['local_fulltext'], 'sha256_pass': True}
    if path.suffix == '.pdf':
        card = path.with_name(path.stem + '_Evidence_Card.md')
        text = card.read_text()
        refs = re.findall(r'\]\(([^)]+\.pdf)\)', text)
        check(bool(refs), '证据卡缺PDF链接: ' + str(card))
        for ref in refs:
            check((card.parent / ref).is_file() and sha(card.parent / ref) == paper['sha256'], '证据卡链接错误: ' + str(card))
        rec['card_pdf_links_match_sha256'] = True
    chain_checks.append(rec)

out = {'time': datetime.now().astimezone().isoformat(), 'root': str(ROOT),
       'boundary': '87条本轮纳入文献的原件身份、路径与文件级结构核对；非全库齐备、全篇精读、独立复现或临床验证',
       'inputs': {'ledger_sha256': sha(A.MANIFEST), 'generator_sha256': sha(ROOT / '06_附录/gen_acquisition_list.py'),
                  'list_sha256': sha(A.OUT), 'audit_script_sha256': sha(Path(__file__))},
       'summary': {'included': len(records), 'unique_identities': len({r['identity'] for r in records}),
                   'needs_fulltext_true': len(records), 'identity_pass': len(records),
                   'formats': dict(Counter(r['format'] for r in records)), 'pending': len(pending),
                   'original_bytes_unchanged': len(records), 'updated_records': len(changes),
                   'changed_fields': dict(field_counts), 'match_methods': dict(method_counts),
                   'direct_index_updated_records': len(index_updates), 'chain_originals_checked': len(chain_checks)},
       'changes': changes, 'index_updates': index_updates, 'chain_checks': chain_checks, 'records': records}
json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
print()
