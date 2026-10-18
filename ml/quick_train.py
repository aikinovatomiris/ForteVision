"""Fast offline ridge baseline using the already installed NumPy/PyTorch stack.
No cloud inference. Test videos are not extracted until model selection is frozen.
"""
import argparse
import json
import random
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
from .dataset import read_labels, sha256, validate_manifest, write_json, ForteError
from .detection import ResNetExtractor
from .preprocessing import SamplingConfig
from .temporal import window_labels, merge_events

class RidgeClassifier:
    def fit(self, x, y, weights, alpha):
        x = np.asarray(x, dtype=np.float64)
        self.mean_ = np.average(x, axis=0, weights=weights)
        self.scale_ = np.sqrt(np.average((x-self.mean_)**2, axis=0, weights=weights)).clip(.01)
        z = (x-self.mean_)/self.scale_
        self.prior_ = float(np.average(y, weights=weights))
        z *= np.sqrt(weights)[:, None]
        target = (y-self.prior_)*np.sqrt(weights)
        if len(z) < z.shape[1]:
            self.coef_ = z.T @ np.linalg.solve(z@z.T + alpha*np.eye(len(z)), target)
        else:
            self.coef_ = np.linalg.solve(z.T@z + alpha*np.eye(z.shape[1]), z.T@target)
        self.n_features_in_ = x.shape[1]
        self.classes_ = np.array([0,1])
        return self
    def predict_proba(self, x):
        x = np.asarray(x)
        if getattr(self, 'context_center_', False):
            x = x - np.median(x, axis=0, keepdims=True)
        score = ((x-self.mean_)/self.scale_)@self.coef_ + self.prior_
        p = 1/(1+np.exp(-np.clip(4*(score-.5),-30,30)))
        return np.column_stack((1-p,p))

def prepare_manifest(root):
    path = root/'artifacts/quick_manifest.json'
    if path.exists(): return path
    grouped = {}
    for row in read_labels(root/'data/raw/open_labels.xlsx', root/'data/videos'):
        h = sha256(root/'data/videos'/row['file'])
        if h not in grouped:
            grouped[h] = dict(row,sha256=h,aliases=[],timestamps=[] if row['timestamp'] is None else [row['timestamp']])
        else:
            old = grouped[h]
            if old['label'] != row['label']: raise ForteError('Conflicting duplicate labels')
            old['aliases'].append(row['file'])
            if row['timestamp'] is not None: old['timestamps']=sorted(set(old['timestamps']+[row['timestamp']]))
    rng = random.Random(42)
    rows = []
    for label in (0,1):
        group = sorted([r for r in grouped.values() if r['label']==label],key=lambda r:r['id'])
        rng.shuffle(group)
        nval = max(1,round(len(group)*.2))
        ntrain = len(group)-2*nval
        for i,row in enumerate(group):
            row['split'] = 'train' if i<ntrain else 'validation' if i<ntrain+nval else 'test'
            rows.append(row)
    manifest=dict(version=1,seed=42,split_method='python_random_stratified_content_grouped',rows=rows)
    validate_manifest(manifest,root/'data/videos')
    write_json(path,manifest)
    return path

def extract(rows,root,extractor,sampling):
    records=[]
    for row in rows:
        start=perf_counter()
        cache=root/('artifacts/quick_cache_'+extractor.image_mode)/f"{row['sha256']}.joblib"
        if cache.exists():
            item=joblib.load(cache)
            if item['weights_sha256']!=extractor.weights_sha256 or item['sampling']!=sampling.to_dict():
                raise ForteError('Cache configuration mismatch')
            x,windows=item['x'],item['windows']
        else:
            print('extract',row['split'],row['id'],flush=True)
            x,windows,info=extractor.extract(root/'data/videos'/row['file'],sampling,start+285)
            if any(t>=info['duration_seconds'] for t in row['timestamps']): raise ForteError('Timestamp outside video')
            cache.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump(dict(x=x,windows=windows,weights_sha256=extractor.weights_sha256,sampling=sampling.to_dict()),cache)
        y,weak=window_labels(windows,row['label'],row['timestamps'],positive_radius=1,exclusion_zone=6)
        records.append(dict(row=row,x=x,windows=windows,y=y,seconds=perf_counter()-start))
    return records

def metrics(records,model,threshold):
    tp=fp=tn=fn=0
    videos=[]
    errors=[]
    for r in records:
        p=model.predict_proba(r['x'])[:,1]
        pred=bool(p.max()>=threshold)
        label=r['row']['label']
        tp+=int(pred and label==1); fp+=int(pred and label==0)
        tn+=int(not pred and label==0); fn+=int(not pred and label==1)
        events=merge_events(r['windows'],p,threshold)
        if label and events:
            errors.extend(min(abs(e['peak']-t) for e in events) for t in r['row']['timestamps'])
        videos.append(dict(id=r['row']['id'],label=label,score=float(p.max()),event_times=[e['peak'] for e in events],extraction_seconds=r['seconds']))
    return dict(f1=2*tp/max(1,2*tp+fp+fn),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),confusion_matrix=[[tn,fp],[fn,tp]],localization_mae_detected_only=float(np.mean(errors)) if errors else None,videos=videos)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--full-frame',action='store_true')
    parser.add_argument('--context',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest=prepare_manifest(root)
    rows=validate_manifest(json.loads(manifest.read_text(encoding='utf-8')),root/'data/videos')
    image_mode='full_frame' if args.full_frame else 'center'
    model_path=root/('models/fullframe_classifier.joblib' if args.full_frame else 'models/quick_classifier.joblib')
    if args.context: model_path=model_path.with_name(model_path.stem+'_context.joblib')
    sampling=SamplingConfig(fps=2,window_seconds=4,stride_seconds=2,batch_size=16)
    extractor=ResNetExtractor(root/'models/resnet18-f37072fd.pth','cpu',image_mode)
    if args.evaluate:
        from .pipeline import load_classifier
        model,meta=load_classifier(model_path)
        if meta['manifest_sha256']!=sha256(manifest): raise ForteError('Manifest changed')
        records=extract([r for r in rows if r['split']=='test'],root,extractor,sampling)
        result=metrics(records,model,meta['threshold'])
        write_json(root/'artifacts'/(model_path.stem+'_test_metrics.json'),result)
        print(json.dumps(result,ensure_ascii=True,indent=2)); return
    if model_path.exists(): raise ForteError('Model exists; refusing to overwrite')
    training=extract([r for r in rows if r['split']=='train'],root,extractor,sampling)
    validation=extract([r for r in rows if r['split']=='validation'],root,extractor,sampling)
    x=np.concatenate([(r['x']-np.median(r['x'],axis=0,keepdims=True) if args.context else r['x'])[r['y']>=0] for r in training])
    y=np.concatenate([r['y'][r['y']>=0] for r in training])
    # Each video contributes equal total mass; then balance the binary classes.
    weights=np.concatenate([np.full(int((r['y']>=0).sum()),1/max(1,int((r['y']>=0).sum()))) for r in training])
    for cls in (0,1): weights[y==cls]*=len(y)/(2*weights[y==cls].sum())
    candidates=[]
    best=None
    for alpha in (10.,100.,1000.):
        model=RidgeClassifier().fit(x,y,weights,alpha)
        model.context_center_=args.context
        scores=np.array([model.predict_proba(r['x'])[:,1].max() for r in validation])
        thresholds=np.unique(np.r_[scores,np.nextafter(scores,1.),.5])
        for threshold in thresholds[(thresholds>0)&(thresholds<1)]:
            report=metrics(validation,model,float(threshold))
            rank=(report['f1'],report['precision'],float(threshold))
            if best is None or rank>best[0]: best=(rank,model,float(threshold),alpha,report)
        candidates.append(dict(alpha=alpha,validation_scores=scores.tolist()))
    _,model,threshold,alpha,report=best
    joblib.dump(model,model_path)
    meta=dict(schema_version=1,architecture='resnet18',pooling='mean_max',embedding_dim=512,feature_dim=1024,
      weights_name='IMAGENET1K_V1',weights_sha256=extractor.weights_sha256,pipeline_sha256=sha256(model_path),
      sampling=sampling.to_dict(),labeling=dict(positive_radius=1,exclusion_zone=6,weak_negatives=False),threshold=threshold,
      seed=42,manifest_sha256=sha256(manifest),train_sha256=[r['sha256'] for r in rows if r['split']=='train'],
      train_ids=[r['id'] for r in rows if r['split']=='train'],validation_ids=[r['id'] for r in rows if r['split']=='validation'],
      classifier='video_balanced_ridge',image_mode=image_mode,context_center=args.context,alpha=alpha,validation_metrics=report,candidates=candidates,test_used_for_training=False,
      limitations='Binary baseline only; scores are not calibrated probabilities; event types and legal mapping not trained.')
    write_json(model_path.with_suffix('.json'),meta)
    print(json.dumps(dict(model=str(model_path),alpha=alpha,threshold=threshold,validation=report),ensure_ascii=True,indent=2))

if __name__=='__main__':
    # Stable import path is required for joblib loading in another process.
    from ml.quick_train import main as run
    run()