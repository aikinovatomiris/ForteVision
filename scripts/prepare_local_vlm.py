"""Explicit one-time downloads. Never invoked by video processing."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def download(item):
    url,path,expected=item
    if path.exists() and expected and hashlib.sha256(path.read_bytes()).hexdigest()==expected:return str(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.download')
    print('Downloading',path.name,flush=True)
    h=hashlib.sha256();size=0;last=0
    with requests.get(url,stream=True,timeout=(20,120)) as r:
        r.raise_for_status()
        with tmp.open('wb') as f:
            for chunk in r.iter_content(4*1024*1024):
                f.write(chunk);h.update(chunk);size+=len(chunk)
                if size-last>256*1024*1024:
                    print(path.name,size//1024//1024,'MiB',flush=True);last=size
    digest=h.hexdigest()
    if expected and digest!=expected: raise ValueError('Hash mismatch '+path.name)
    tmp.replace(path)
    print('Verified',path.name,size,flush=True)
    return dict(file=str(path.relative_to(ROOT)),url=url,sha256=digest,size=size)

def main():
    repo='ggml-org/Qwen3-VL-2B-Instruct-GGUF'
    d=requests.get('https://huggingface.co/api/models/'+repo,timeout=30).json();revision=d['sha']
    tree=requests.get('https://huggingface.co/api/models/'+repo+'/tree/'+revision,timeout=30).json()
    items=[]
    for f in tree:
        if f['path'].endswith('.gguf'):
            items.append(('https://huggingface.co/'+repo+'/resolve/'+revision+'/'+f['path'],ROOT/'models/vlm'/f['path'],f['lfs']['oid']))
    release=requests.get('https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b11716',timeout=30).json()
    for asset in release['assets']:
        if asset['name'] in ['llama-b11716-bin-win-cuda-12.4-x64.zip','cudart-llama-bin-win-cuda-12.4-x64.zip']:
            digest=asset.get('digest','')
            items.append((asset['browser_download_url'],ROOT/'runtime/downloads'/asset['name'],digest.removeprefix('sha256:') or None))
    with ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(download,items))
    dest=ROOT/'runtime/llama-b11716';dest.mkdir(parents=True,exist_ok=True)
    for _,path,_ in items:
        if path.suffix=='.zip':
            with zipfile.ZipFile(path) as z:
                for member in z.infolist():
                    if not (dest/member.filename).resolve().is_relative_to(dest.resolve()):raise ValueError('Unsafe zip path')
                z.extractall(dest)
    (ROOT/'models/vlm/provenance.json').write_text(json.dumps(dict(repository=repo,revision=revision,original='Qwen/Qwen3-VL-2B-Instruct',license='Apache-2.0',runtime_release='b11716',files=results),indent=2),encoding='utf-8')
    print('Local VLM files ready',flush=True)
if __name__=='__main__':main()