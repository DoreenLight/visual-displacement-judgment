"""Regression tests for the user-facing *missing fulltexts only* list."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('acquisition', Path(__file__).parents[1] / 'gen_acquisition_list.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def paper(**changes):
    return {'id': 'test', 'title': 'A verified rehabilitation paper', 'doi': '10.1234/test',
            'screening': 'included', 'needs_fulltext': True, 'kind': 'paper',
            'venue': 'Verified Journal', 'database': 'Verified Database',
            'landing_url': 'https://doi.org/10.1234/test', **changes}


def manifest(papers):
    return {'schema': 'rehab-chain-research/v1', 'papers': papers}


def xml(path, doi='10.1234/test', title='A verified rehabilitation paper', body='<p>Full text.</p>'):
    path.write_text('<article><front><article-meta><article-id pub-id-type="doi">'
                    + doi + '</article-id><title-group><article-title>' + title
                    + '</article-title></title-group></article-meta></front><body>'
                    + body + '</body></article>')


class DownloadOnlyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def pending(self, papers, catalog=None, unindexed=None):
        return A.pending_papers(manifest(papers), self.root,
                                catalog=catalog or {}, unindexed=unindexed or [])

    def test_only_included_required_papers(self):
        candidates = [paper(), paper(doi='10.1234/skip', screening='excluded'),
                      paper(doi='10.1234/web', kind='web'),
                      paper(doi='10.1234/optional', needs_fulltext=False)]
        self.assertEqual(len(self.pending(candidates)), 1)

    def test_duplicate_identity_is_one_row(self):
        self.assertEqual(len(self.pending([paper(), paper(doi='https://doi.org/10.1234/TEST')])), 1)

    def test_cross_library_fulltext_is_not_download_task(self):
        p = self.root / 'cross-library.xml'
        xml(p)
        self.assertEqual(self.pending([paper()], {A.identity(paper()): [p]}), [])

    def test_unindexed_xml_is_found_by_identity(self):
        p = self.root / 'manual.xml'
        xml(p)
        self.assertEqual(self.pending([paper()], unindexed=[p]), [])

    def test_bibliography_doi_is_not_document_identity(self):
        p = self.root / 'different-paper.xml'
        xml(p, doi='10.1234/other', title='Unrelated article',
            body='<ref><pub-id pub-id-type="doi">10.1234/test</pub-id></ref>')
        self.assertEqual(len(self.pending([paper()], unindexed=[p])), 1)

    def test_missing_declared_file_stays_pending(self):
        self.assertEqual(len(self.pending([paper(local_fulltext='missing.pdf')])), 1)

    def test_wrong_old_index_identity_does_not_hide_actual_fulltext(self):
        import json
        folder = self.root / '01_参考论文/corpus'
        folder.mkdir(parents=True)
        file = folder / 'actual.xml'
        xml(file)
        old = paper(doi='10.1234/wrong', title='Wrong old metadata',
                    local_fulltext='01_参考论文/corpus/actual.xml')
        (folder / 'index.json').write_text(json.dumps({'papers': [old]}))
        self.assertEqual(A.pending_papers(manifest([paper()]), self.root), [])

    def test_duplicate_record_with_file_removes_whole_group(self):
        p = self.root / 'existing.xml'
        xml(p)
        self.assertEqual(self.pending([paper(), paper(local_fulltext=p.name)]), [])

    def test_xml_without_body_is_not_fulltext(self):
        p = self.root / 'abstract-only.xml'
        p.write_text('<article><article-meta><article-id pub-id-type="doi">10.1234/test'
                     '</article-id></article-meta></article>')
        self.assertEqual(len(self.pending([paper()], unindexed=[p])), 1)

    def test_file_outside_root_is_not_accepted(self):
        self.assertIsNone(A.local_path(self.root, '../elsewhere.pdf'))

    def test_unsupported_manifest_fails_closed(self):
        with self.assertRaises(ValueError):
            A.pending_papers({'schema': 'wrong', 'papers': []}, self.root, {}, [])

    def test_invalid_address_fails_closed(self):
        with self.assertRaises(ValueError):
            self.pending([paper(landing_url='javascript:alert(1)')])

    def test_empty_list_contains_no_history(self):
        text = A.render([])
        self.assertEqual(text, '# 需要你下载的论文\n\n当前无待下载论文。\n')

    def test_required_columns_only(self):
        text = A.render(self.pending([paper()]))
        self.assertIn('论文标题 | 地址 | 期刊/会议 | 数据库/平台', text)
        for old in ['已入库', 'P0', '网页型', '未命中', '§']:
            self.assertNotIn(old, text)

    def test_explicit_unknown_fields(self):
        p = self.pending([paper(venue='', database='')])[0]
        self.assertEqual(p['venue'], '未核实')
        self.assertEqual(p['database'], '未核实')

    def test_chinese_titles_not_empty_identity(self):
        self.assertTrue(A.normalize_title('康复研究'))

    def test_pdf_layout_retry_recovers_title_reading_order(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        p = self.root / 'reading-order.pdf'
        p.write_bytes(b'%PDF-1.7\nfixture')

        def extract(args, **kwargs):
            text = ('A verified rehabilitation paper' if '-layout' in args
                    else 'rehabilitation paper\nJournal header\nA verified')
            return SimpleNamespace(stdout=text)

        with patch.object(A.subprocess, 'run', side_effect=extract):
            self.assertEqual(self.pending([paper()], unindexed=[p]), [])

    def test_pdf_layout_retry_does_not_accept_doi_alone(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        p = self.root / 'wrong-title.pdf'
        p.write_bytes(b'%PDF-1.7\nfixture')
        with patch.object(A.subprocess, 'run', return_value=SimpleNamespace(
                stdout='An unrelated article\nReferences: 10.1234/test')):
            self.assertEqual(len(self.pending([paper()], unindexed=[p])), 1)

    def test_verified_accepted_manuscript_title_alias(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        p = self.root / 'author-version.pdf'
        p.write_bytes(b'%PDF-1.7\nfixture')
        item = paper(title='Published formal title', title_aliases=['Accepted manuscript title'])
        with patch.object(A.subprocess, 'run', return_value=SimpleNamespace(stdout='Accepted manuscript title')):
            self.assertTrue(A.content_matches(p, item))
            self.assertFalse(A.content_matches(p, paper(title='Published formal title')))


if __name__ == '__main__':
    unittest.main()
