#!/usr/bin/env python3
"""Exercise the actual read-only Case 1/2 blocks against URI/raw-path fixtures."""
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote, unquote
import re

SCRIPT = Path(__file__).with_name('verify_kb.sh').read_text(encoding='utf-8')
CASE1 = SCRIPT.split('# ---------------------------------------------------------------- Case 1\n', 1)[1].split('# ---------------------------------------------------------------- Case 2\n', 1)[0]
CASE2 = SCRIPT.split('# ---------------------------------------------------------------- Case 2\n', 1)[1].split('# ---------------------------------------------------------------- Case 3\n', 1)[0]


class RelativeLinks(unittest.TestCase):
    def run_case(self, block, filename, target, actual=None, image=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            if actual is not None:
                (root / actual).write_bytes(b'fixture')
            (root / filename).write_text(('!' if image else '') + '[target](' + target + ')\n')
            namespace = {'ROOT': root, 'excluded': lambda _: False, 'cases': [],
                         're': re, 'unquote': unquote}
            exec(compile(block, 'verify_kb.sh::link-case', 'exec'), namespace)
            return namespace['cases'][0][2]

    def test_encoded_unicode_spaces_and_fragment(self):
        actual = '临床记录 #1.txt'
        self.assertEqual(self.run_case(CASE1, 'entry.md', quote(actual) + '#section', actual), 0)

    def test_raw_existing_path_still_supported(self):
        self.assertEqual(self.run_case(CASE1, 'entry.md', 'raw notes.txt', 'raw notes.txt'), 0)

    def test_missing_encoded_target_still_fails(self):
        self.assertEqual(self.run_case(CASE1, 'entry.md', quote('缺失文件.txt')), 1)

    def test_percent_decoded_once_only(self):
        actual = 'notes%20draft.txt'
        self.assertEqual(self.run_case(CASE1, 'entry.md', quote(actual), actual), 0)
        self.assertEqual(self.run_case(CASE1, 'entry.md', actual, actual), 1)

    def test_encoded_image_path(self):
        actual = '指标 图.png'
        self.assertEqual(self.run_case(CASE2, 'example_全文翻译.md', quote(actual), actual, image=True), 0)

    def test_missing_image_still_fails(self):
        self.assertEqual(self.run_case(CASE2, 'example_全文翻译.md', quote('缺图.png'), image=True), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
