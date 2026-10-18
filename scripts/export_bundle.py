"""Export tracked source and active model locally; no footage, secrets or job data."""
import argparse,json,subprocess,zipfile
from pathlib import Path
from ml.dataset import sha256

def allowed_source(name):
    p=Path(name)
    return (not any(part in ('.git','.venv','.venv312','data','artifacts','models','runtime','__pycache__') for part in p.parts)
            and (not p.name.startswith('.env') or p.name=='.env.example')
            and p.suffix.lower() not in ('.mov','.mp4','.m4v','.avi','.sqlite3','.whl','.gguf','.joblib','.pth','.png','.jpg','.jpeg')
            and p.name!='secrets.toml')

def main():
    from app.config import Settings,ROOT
    from ml.pipeline import load_classifier
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    settings=Settings();_,meta=load_classifier(settings.model)
    if sha256(settings.weights)!=meta['weights_sha256']:raise ValueError('Backbone hash mismatch')
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0')
    files={name:ROOT/name for name in names if name and allowed_source(name)}
    for p in (settings.model,settings.model.with_suffix('.json'),settings.weights):files['models/'+p.name]=p
    # The example must select the actual bundled artifact; no local .env is copied.
    example=(ROOT/'.env.example').read_text(encoding='utf-8')
    if f'FORTEVISION_MODEL=models/{settings.model.name}' not in example:raise ValueError('Example model differs from exported model')
    manifest=dict(schema_version=1,model=settings.model.name,review_mode=True,
                  files={n:sha256(p) for n,p in sorted(files.items())})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',zipfile.ZIP_DEFLATED) as archive:
        for name,path in sorted(files.items()):archive.write(path,'ForteVision/'+name)
        archive.writestr('ForteVision/bundle_manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip():raise ValueError('ZIP integrity failure')
    print(json.dumps(dict(bundle=str(args.output),files=len(files),sha256=sha256(args.output))))

if __name__=='__main__':main()
