import pytest
from scripts.init_env import initialize
from scripts.export_bundle import allowed_source

def test_init_preserves_existing_key(tmp_path):
    (tmp_path/'.env.example').write_text('FORTEVISION_API_KEY=replace-with-a-long-random-local-secret')
    (tmp_path/'.env').write_text('already-configured')
    with pytest.raises(FileExistsError):initialize(tmp_path)
    assert (tmp_path/'.env').read_text()=='already-configured'

def test_init_generates_distinct_private_keys(tmp_path):
    for name in ('a','b'):
        p=tmp_path/name;p.mkdir();(p/'.env.example').write_text('FORTEVISION_API_KEY=replace-with-a-long-random-local-secret');initialize(p)
    a=(tmp_path/'a/.env').read_text();b=(tmp_path/'b/.env').read_text()
    assert a!=b and 'replace-with-' not in a

@pytest.mark.parametrize('name',['.env','.env.production','data/video.mov','artifacts/result.json','models/unrelated.joblib','.streamlit/secrets.toml','runtime/server.exe','client.jpg','installer.whl'])
def test_export_excludes_private_runtime_files(name):
    assert not allowed_source(name)

def test_export_keeps_public_setup_template():
    assert allowed_source('.env.example') and allowed_source('app/api.py')
