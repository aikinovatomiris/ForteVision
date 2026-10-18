"""Evaluate cropped features with the same grouped selection policy as the full-frame candidate."""
import json,dataclasses,time
from pathlib import Path
import joblib,numpy as np
from ml.multiscale import MultiScaleExtractor
from ml.preprocessing import SamplingConfig
from ml.dataset import key,sha256,write_json
from scripts import train_generalized as training

ROOT=Path(__file__).resolve().parents[1]
CONFIG=SamplingConfig(fps=.5,window_seconds=4,stride_seconds=2,batch_size=20)

def main():
    rows=training.annotated_rows()
    extractor=MultiScaleExtractor(ROOT/'models/resnet18-f37072fd.pth')
    cache=ROOT/'artifacts/multiscale_cache';cache.mkdir(exist_ok=True)
    files={key(p.stem):p for p in (ROOT/'data/videos').iterdir() if p.suffix.lower()=='.mov'}
    for i,r in enumerate(rows):
        p=cache/(r['row']['sha256']+'.joblib')
        if p.exists():
            d=joblib.load(p)
            if d['sampling']!=dataclasses.asdict(CONFIG) or d['weights_sha256']!=extractor.weights_sha256:raise ValueError('Cache mismatch')
        else:
            path=files[r['row']['id']]
            if sha256(path)!=r['row']['sha256']:raise ValueError('Video content mismatch')
            start=time.perf_counter();x,windows,info=extractor.extract(path,CONFIG)
            d=dict(x=x,windows=windows,sampling=dataclasses.asdict(CONFIG),weights_sha256=extractor.weights_sha256);joblib.dump(d,p)
            print('extracted',i+1,'/',len(rows),r['row']['id'],round(time.perf_counter()-start,1),'seconds',flush=True)
        # Transfer eligible time intervals, never ordinal window assumptions.
        intervals=[(w['start'],w['end']) for j,w in enumerate(r['windows']) if r['eligible'][j]]
        r['eligible']=np.array([any(a<=w['timestamp']<b for a,b in intervals) for w in d['windows']])
        r['x']=d['x'];r['windows']=d['windows']
        if not r['eligible'].any():raise ValueError('No eligible window')
    training.main(rows=rows,model_name='multiscale_generalized_v1',report_name='multiscale_evaluation',fallback_name='multiscale_typed_v1',image_mode='multiscale',sampling=dataclasses.asdict(CONFIG))

if __name__=='__main__':main()
