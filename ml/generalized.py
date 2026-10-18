"""Separate binary temporal detection from tentative event naming.

No video identifiers, absolute timestamps or reference-video lookup at inference.
"""
import numpy as np
from scipy.special import expit
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .typed import CATALOG
from .temporal import merge_events


def features(x,mode):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=1024 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty windows x 1024')
    mean=x[:,:512]
    norm=np.linalg.norm(mean,axis=1,keepdims=True).clip(1e-8)
    appearance=mean/norm
    variation=(x[:,512:]-mean)/norm
    if mode=='appearance':return np.concatenate((appearance,variation),axis=1)
    median=np.median(appearance,axis=0,keepdims=True)
    past=np.array([appearance[max(0,i-3):i+1].mean(0) for i in range(len(x))])
    future=np.array([appearance[i:min(len(x),i+4)].mean(0) for i in range(len(x))])
    relative=np.concatenate((appearance-median,variation,appearance-past,future-appearance),axis=1)
    if mode=='relative':return relative
    if mode=='combined':return np.concatenate((appearance,relative),axis=1)
    raise ValueError('Unknown feature mode')


def fit_binary(rows,mode,alpha):
    transformed=[features(r['x'],mode) for r in rows]
    indices=[np.flatnonzero(r['eligible']) for r in rows]
    for iteration in range(2):
        x=np.concatenate([z[idx] for z,idx in zip(transformed,indices)])
        y=np.concatenate([np.repeat(int(r['label']!='normal'),len(idx)) for r,idx in zip(rows,indices)])
        weight=np.concatenate([np.full(len(idx),1/len(idx)) for idx in indices]);weight*=len(weight)/weight.sum()
        model=make_pipeline(StandardScaler(),RidgeClassifier(alpha=alpha,class_weight='balanced',solver='lsqr',tol=.002))
        model.fit(x,y,ridgeclassifier__sample_weight=weight)
        for i,r in enumerate(rows):
            if not r['weak']:continue
            score=model.decision_function(transformed[i]);width=min(3,len(score))
            start=int(np.convolve(score,np.ones(width)/width,'valid').argmax())
            indices[i]=np.arange(start,start+width)
    return model


class GeneralizedClassifier:
    classes_=np.array([0,1])
    n_features_in_=1024
    def __init__(self,detector,mode,typed_head):
        self.detector=detector;self.mode=mode;self.typed_head=typed_head
    def predict_proba(self,x):
        p=expit(self.detector.decision_function(features(x,self.mode)))
        return np.column_stack((1-p,p))
    def events(self,x,windows,threshold):
        score=self.predict_proba(x)[:,1]
        category=self.typed_head.category_probabilities(x)
        labels=list(self.typed_head.estimator.classes_)
        result=[]
        for event in merge_events(windows,score,threshold):
            positions=[i for i,w in enumerate(windows) if w['start']<event['end'] and w['end']>event['start']]
            index=max(positions,key=lambda i:score[i])
            winner=int(category[index].argmax());name=labels[winner]
            if name!='normal' and category[index,winner]>=.55:
                title,description=CATALOG[name]
                event.update(event_type=name,type=title,description='Предполагаемый тип по визуальному сходству: '+description)
            else:
                event.update(type='Тип события не определён',description='Временной детектор выделил подозрительный фрагмент, но классификатор не смог надёжно определить тип. Просмотрите последовательность; конкретное нарушение и норматив не установлены.')
            event['description']+=' Оценка учитывает изменения до и после фрагмента; это гипотеза, а не подтверждённое нарушение.'
            result.append(event)
        return result
