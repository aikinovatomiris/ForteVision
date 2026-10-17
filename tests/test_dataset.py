import json
import tempfile
import unittest
from datetime import time
from pathlib import Path

from ml.dataset import ForteError, key, read_labels, timestamp, validate_manifest, sha256


class DatasetTests(unittest.TestCase):
    def test_filename_normalization(self):
        self.assertEqual(key('  Х-1.MOV '), key('X-1'))
        self.assertEqual(key('Ｘ-1.mov'), key('x-1.mov'))

    def test_timestamps(self):
        for value, expected in [('22 секунда', 22), ('1минута 8 секунда', 68), ('01:08', 68), ('1:02:03', 3723), (time(0, 1, 8), 68), (0, 0), (None, None)]:
            with self.subTest(value=value):
                self.assertEqual(timestamp(value), expected)

    def test_invalid_timestamps(self):
        for value in ('tomorrow', '1:99', -1, float('nan')):
            with self.subTest(value=value), self.assertRaises(ForteError):
                timestamp(value)

    def test_read_labels_and_ambiguity(self):
        from openpyxl import Workbook
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'Х-1.mov').write_bytes(b'fixture')
            book = Workbook()
            book.active.append(['наименование файла', 'таймстемп', 'нарушение / без нарушения'])
            book.active.append(['X-1', '1минута 8 секунда', 'нарушение'])
            book.save(root/'labels.xlsx')
            self.assertEqual(read_labels(root/'labels.xlsx', root)[0]['timestamp'], 68)
            (root/'X-1.mov').write_bytes(b'ambiguous')
            with self.assertRaisesRegex(ForteError, 'Неоднозначные'):
                read_labels(root/'labels.xlsx', root)

    def test_missing_labels(self):
        with self.assertRaises(ForteError):
            read_labels(Path('not-present.xlsx'), Path('.'))

    def test_manifest_hash_and_cross_split_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for split in ('train', 'validation', 'test'):
                for label in (0, 1):
                    name = f'{split}-{label}.mov'
                    (root/name).write_bytes(name.encode())
                    rows.append(dict(id=key(name), file=name, sha256=sha256(root/name), split=split, label=label, timestamp=1 if label else None))
            manifest = dict(version=1, rows=rows)
            validate_manifest(manifest, root)
            (root/'test-1.mov').write_bytes(b'changed')
            with self.assertRaisesRegex(ForteError, 'изменилось'):
                validate_manifest(manifest, root)
            (root/'test-1.mov').write_bytes((root/'train-1.mov').read_bytes())
            rows[-1]['sha256'] = sha256(root/'test-1.mov')
            with self.assertRaisesRegex(ForteError, 'одинаковые'):
                validate_manifest(manifest, root)

    def test_split_is_by_video_and_reproducible(self):
        try:
            import sklearn
        except ImportError as exc:
            self.skipTest(str(exc))
        from openpyxl import Workbook
        from ml.dataset import prepare
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            book = Workbook()
            book.active.append(['наименование файла', 'таймстемп', 'нарушение / без нарушения'])
            for i in range(30):
                (root/f'X-{i}.mov').write_bytes(str(i).encode())
                book.active.append([f'Х-{i}', 1 if i % 2 else None, i % 2])
            (root/'X-99.mov').write_bytes((root/'X-1.mov').read_bytes())
            book.active.append(['X-99', 2, 1])
            book.save(root/'labels.xlsx')
            prepare(root/'labels.xlsx', root, root/'a.json')
            prepare(root/'labels.xlsx', root, root/'b.json')
            a, b = json.loads((root/'a.json').read_text()), json.loads((root/'b.json').read_text())
            self.assertEqual(a, b)
            self.assertEqual(a['source_rows'],31)
            self.assertEqual(a['unique_videos'],30)
            combined = next(r for r in a['rows'] if r['id']=='x-1')
            self.assertEqual(combined['timestamps'],[1,2])
            self.assertEqual(combined['aliases'],['X-99.mov'])
            groups = [{r['id'] for r in a['rows'] if r['split'] == split} for split in ('train', 'validation', 'test')]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
