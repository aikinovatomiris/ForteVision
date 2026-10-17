import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ml.dataset import ForteError
from ml.temporal import merge_events, window_labels


class TemporalTests(unittest.TestCase):
    def test_weak_negatives_exclusion(self):
        windows = [dict(start=s, end=s+4, timestamp=s+2) for s in (0, 8, 10, 20)]
        labels, weak = window_labels(windows, 1, 11, positive_radius=0, exclusion_zone=5, weak_negatives=True)
        self.assertEqual(labels.tolist(), [0, 1, 1, 0])
        self.assertEqual(weak.tolist(), [True, False, False, True])
        labels, _ = window_labels(windows, 1, 11, positive_radius=0, exclusion_zone=5)
        self.assertEqual(labels.tolist(), [-1, 1, 1, -1])

    def test_merging_and_peak(self):
        windows = [dict(start=s, end=s+4, timestamp=s+1.5) for s in (0, 2, 4, 12)]
        events = merge_events(windows, [.6, .9, .2, .8], .5)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['peak'], 3.5)
        self.assertEqual(events[0]['end'], 6)
        self.assertEqual(merge_events(windows, [0, 0, 0, 0], .5), [])

    def test_all_annotations_protect_against_weak_negatives(self):
        windows = [dict(start=56,end=60), dict(start=66,end=70), dict(start=90,end=94)]
        y, weak = window_labels(windows, 1, [57,68], positive_radius=0, exclusion_zone=5, weak_negatives=True)
        self.assertEqual(y.tolist(), [1,1,0])
        self.assertEqual(weak.tolist(), [False,False,True])

    def test_empty_rules_and_generic_prohibition(self):
        from ml.rules import GENERIC_TYPE, RuleEngine
        self.assertEqual(RuleEngine().sources_for(GENERIC_TYPE), [])
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'rules.json'
            path.write_text(json.dumps({'rules': [{'event_type': GENERIC_TYPE}]}))
            with self.assertRaises(ForteError):
                RuleEngine(path)


class VideoTests(unittest.TestCase):
    def setUp(self):
        try:
            import cv2
            self.cv2 = cv2
        except ImportError as exc:
            self.skipTest(str(exc))
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)/'synthetic.avi'
        writer = self.cv2.VideoWriter(str(self.path), self.cv2.VideoWriter_fourcc(*'MJPG'), 10, (96, 64))
        self.assertTrue(writer.isOpened())
        for i in range(85):
            frame = np.zeros((64,96,3), np.uint8)
            frame[:, i%70:i%70+20] = 180
            writer.write(frame)
        writer.release()

    def test_video_timestamps_pooling_and_timeout(self):
        from ml.preprocessing import SamplingConfig, iter_frames, pooled_windows, probe
        config = SamplingConfig(fps=2, window_seconds=4, stride_seconds=2)
        samples = list(iter_frames(self.path, config))
        times = [t for t, _ in samples]
        self.assertEqual(len(samples), 17)
        self.assertTrue(all(b > a for a,b in zip(times, times[1:])))
        self.assertAlmostEqual(probe(self.path)['duration_seconds'], 8.5)
        x, windows = pooled_windows(times, np.ones((len(times),512)), config, 8.5)
        self.assertEqual(x.shape[1], 1024)
        self.assertTrue(np.all(x == 1))
        with self.assertRaises(ForteError):
            list(iter_frames(self.path, config, deadline=0))

    def test_evidence_is_real_jpeg(self):
        from ml.evidence import evidence_frames
        frames = evidence_frames(self.path, dict(start=1, end=4, peak=2), count=3)
        self.assertEqual(len(frames), 3)
        for frame in frames:
            raw = base64.b64decode(frame['image_base64'], validate=True)
            self.assertTrue(raw.startswith(b'\xff\xd8'))
            self.assertIsNotNone(self.cv2.imdecode(np.frombuffer(raw, np.uint8), self.cv2.IMREAD_COLOR))
            self.assertTrue(1 <= frame['timestamp_sec'] <= 4)

    def test_evidence_does_not_depend_on_seek(self):
        from ml.preprocessing import read_at_times
        original_capture = self.cv2.VideoCapture
        class NoSeek:
            def __init__(self, *args): self.cap = original_capture(*args)
            def __getattr__(self, name): return getattr(self.cap, name)
            def set(self, *args): raise AssertionError('Seek is not reliable for all MOVs')
        with patch('ml.preprocessing.cv2.VideoCapture', NoSeek):
            frames = read_at_times(self.path, [1.02, 2.05, 4.1])
        self.assertEqual(len(frames), 3)
        self.assertAlmostEqual(frames[0][0], 1.0)

    def test_negative_preroll_is_skipped_without_shifting_clock(self):
        from ml.preprocessing import SamplingConfig, iter_frames
        original_capture = self.cv2.VideoCapture
        cv2 = self.cv2
        class Preroll:
            def __init__(self, *args):
                self.cap, self.reads = original_capture(*args), 0
            def __getattr__(self, name): return getattr(self.cap, name)
            def read(self):
                self.reads += 1
                return self.cap.read()
            def get(self, prop):
                if prop == cv2.CAP_PROP_POS_MSEC and self.reads == 1:
                    return -100.0
                return self.cap.get(prop)
        with patch('ml.preprocessing.cv2.VideoCapture', Preroll):
            frames = list(iter_frames(self.path, SamplingConfig()))
        self.assertAlmostEqual(frames[0][0], .1)

    def test_corrupt_video(self):
        from ml.preprocessing import probe
        self.path.write_bytes(b'broken')
        with self.assertRaises(ForteError):
            probe(self.path)

    def test_missing_weights_do_not_download(self):
        from ml.detection import ResNetExtractor
        with patch('urllib.request.urlopen', side_effect=AssertionError('No network')):
            with self.assertRaisesRegex(ForteError, 'Нет локальных весов'):
                ResNetExtractor(self.path.with_suffix('.pth'))


class ModelTests(unittest.TestCase):
    def setUp(self):
        try:
            from ml.pipeline import load_classifier
            self.load_classifier = load_classifier
        except ImportError as exc:
            self.skipTest(str(exc))

    def test_missing_model(self):
        with self.assertRaisesRegex(ForteError, 'не обучен'):
            self.load_classifier(Path('absent-model.joblib'))

    def test_train_never_extracts_test_videos(self):
        import json
        import types
        from ml.dataset import key, sha256
        from ml.preprocessing import SamplingConfig
        from ml.training import train
        calls = []
        class Extractor:
            weights_sha256 = 'f37072fd'+'0'*56
            torch = types.SimpleNamespace(manual_seed=lambda seed: None)
            def __init__(self, *args): pass
            def extract(self, path, *args):
                calls.append(path.name)
                label = int(path.stem[-1])
                return np.full((2,1024), label, dtype=float), [dict(start=0,end=4,timestamp=2), dict(start=6,end=10,timestamp=8)], dict(duration_seconds=10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for split in ('train','validation','test'):
                for label in (0,1):
                    name = f'{split}-{label}.mov'
                    (root/name).write_bytes(name.encode())
                    rows.append(dict(id=key(name),file=name,split=split,label=label,timestamp=1 if label else None,sha256=sha256(root/name)))
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(dict(version=1,rows=rows)))
            fake_torch = types.SimpleNamespace(__version__='unit-test')
            with patch('ml.training.ResNetExtractor', Extractor), patch.dict('sys.modules', torch=fake_torch, torchvision=fake_torch):
                metadata = train(manifest, root, root/'model.joblib', root/'unused.pth', SamplingConfig(), dict(positive_radius=0,exclusion_zone=5,weak_negatives=False))
            self.assertFalse(any(name.startswith('test') for name in calls))
            self.assertEqual(metadata['training_windows'],3)
            self.assertFalse(metadata['test_used_for_training'])
            self.load_classifier(root/'model.joblib')

    def test_threshold_uses_validation_only_and_reports_missing_classes(self):
        from ml.training import binary_metrics, tune_threshold
        records = [{'row': {'label': 0}}, {'row': {'label': 1}}]
        threshold = tune_threshold(records, [np.array([.2]), np.array([.8])])
        self.assertTrue(.2 < threshold <= .8)
        with self.assertRaises(ForteError):
            tune_threshold(records[:1], [np.array([.2])])
        self.assertIsNone(binary_metrics([0], [.2], .5)['roc_auc'])

    def test_pipeline_contract_with_mock_model_and_mock_evidence(self):
        # Unit tests never create the delivered classifier or download pretrained weights.
        from ml.pipeline import analyze
        class Classifier:
            def predict_proba(self, x):
                return np.array([[.1,.9]])
        class Extractor:
            weights_sha256 = 'fixture'
            def __init__(self, *args): pass
            def extract(self, *args):
                return np.zeros((1,1024)), [dict(start=0,end=4,timestamp=2)], {}
        metadata = dict(sampling={}, weights_sha256='fixture', threshold=.5)
        with patch('ml.pipeline.load_classifier', return_value=(Classifier(),metadata)), patch('ml.pipeline.ResNetExtractor', Extractor), patch('ml.pipeline.evidence_frames', return_value=[dict(timestamp_sec=2,image_base64='unit-test-placeholder')]):
            result = analyze(Path('fixture.mov'), Path('fixture.joblib'), Path('fixture.pth'))
        self.assertEqual(result['verdict'], 'violation')
        self.assertEqual(result['violations'][0]['timecode_sec'], 2)
