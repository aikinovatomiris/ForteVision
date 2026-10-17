import tempfile
import threading
import time
import unittest
from pathlib import Path

MOV_HEADER = b'\x00\x00\x00\x18ftypqt  '+b'\x00'*16


class ApiTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from app.api import create_app
            from app.config import Settings
        except ImportError as exc:
            self.skipTest(str(exc))
        self.Client, self.create_app, self.Settings = TestClient, create_app, Settings
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = Settings(api_key='unit-test-key', upload_dir=self.root/'uploads', model=self.root/'missing.joblib', weights=self.root/'missing.pth', max_upload_mb=1, queue_size=1)
        self.headers = {'X-API-Key': 'unit-test-key'}

    def upload(self, client, name='sample.mov', content=MOV_HEADER, headers=None):
        return client.post('/api/v1/analyze', files={'file': (name,content,'video/quicktime')}, headers=self.headers if headers is None else headers)

    def wait_job(self, client, job_id):
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            response = client.get('/api/v1/jobs/'+job_id, headers=self.headers)
            self.assertEqual(response.status_code,200)
            if response.json()['status'] != 'processing':
                return response.json()
            time.sleep(.01)
        self.fail('Job did not complete')

    def test_authentication_formats_and_size(self):
        with self.Client(self.create_app(self.settings, lambda path: {})) as client:
            self.assertEqual(client.get('/health').status_code,200)
            self.assertEqual(self.upload(client, headers={}).status_code,401)
            self.assertEqual(self.upload(client, name='x.mp4').status_code,415)
            self.assertEqual(self.upload(client, content=b'broken').status_code,422)
            self.assertEqual(self.upload(client, content=MOV_HEADER+b'x'*(1024*1024)).status_code,413)
            self.assertEqual(list(self.settings.upload_dir.iterdir()),[])

    def test_done_schema_safe_names_cleanup(self):
        seen = []
        def runner(path):
            seen.append(path)
            return dict(verdict='no_violation', violations=[], elapsed_seconds=.01)
        with self.Client(self.create_app(self.settings, runner)) as client:
            response = self.upload(client, name='../../escape.mov')
            self.assertEqual(response.status_code,202)
            self.assertEqual(set(response.json()), {'job_id'})
            job = self.wait_job(client,response.json()['job_id'])
            self.assertEqual(job['status'],'done')
            self.assertEqual(job['verdict'],'no_violation')
            self.assertEqual(job['violations'],[])
            self.assertEqual(seen[0].parent,self.settings.upload_dir)
            self.assertFalse(seen[0].exists())

    def test_processing_queue_full_and_failed(self):
        started, release = threading.Event(), threading.Event()
        def runner(path):
            started.set()
            release.wait(3)
            raise RuntimeError('Классификатор не обучен')
        with self.Client(self.create_app(self.settings, runner)) as client:
            try:
                first = self.upload(client).json()['job_id']
                self.assertTrue(started.wait(1))
                self.assertEqual(client.get('/api/v1/jobs/'+first,headers=self.headers).json()['status'],'processing')
                self.assertEqual(self.upload(client).status_code,202)
                self.assertEqual(self.upload(client).status_code,429)
            finally:
                release.set()
            result = self.wait_job(client,first)
            self.assertEqual(result['status'],'failed')
            self.assertNotIn('verdict',result)
            self.assertIn('не обучен',result['error'])

    def test_missing_model_actual_worker(self):
        try:
            from ml.pipeline import load_classifier
        except ImportError as exc:
            self.skipTest(str(exc))
        def runner(path):
            load_classifier(self.settings.model)
        with self.Client(self.create_app(self.settings, runner)) as client:
            job_id = self.upload(client).json()['job_id']
            self.assertEqual(self.wait_job(client,job_id)['status'],'failed')

    def test_schema_rejects_inconsistent_verdict(self):
        from app.schemas import DoneJob
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            DoneJob(job_id='x', status='done', verdict='violation', violations=[])

    def test_subprocess_timeout_cleans_partial_result(self):
        import subprocess
        from unittest.mock import patch
        from app.jobs import subprocess_runner
        path = self.root/'input.mov'
        partial = path.with_suffix('.result.json.tmp')
        partial.write_text('partial')
        with patch('app.jobs.subprocess.run', side_effect=subprocess.TimeoutExpired('worker',1)):
            with self.assertRaisesRegex(RuntimeError,'процесс остановлен'):
                subprocess_runner(path,self.settings)
        self.assertFalse(partial.exists())
