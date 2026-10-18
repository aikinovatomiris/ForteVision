"""Order-sensitive temporal model. Rule steps are not inferred from a class label."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml.typed import TypedClassifier

def sequence_features(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=1024 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty (windows,1024) features')
    # 4-second windows sampled every 2 seconds. No filename or absolute position.
    # Mean component is appearance; max-minus-mean approximates within-window variation.
    appearance=x[:,:512]
    norm=np.linalg.norm(appearance,axis=1,keepdims=True).clip(1e-8)
    appearance=appearance/norm
    dynamic=(x[:,512:]-x[:,:512])/norm
    baseline=np.median(appearance,axis=0,keepdims=True)
    previous=[];following=[];short_previous=[];short_following=[]
    for i in range(len(x)):
        previous.append(appearance[max(0,i-6):i].mean(axis=0) if i else appearance[i])
        following.append(appearance[i+1:min(len(x),i+7)].mean(axis=0) if i+1<len(x) else appearance[i])
        short_previous.append(appearance[max(0,i-2)])
        short_following.append(appearance[min(len(x)-1,i+2)])
    previous=np.asarray(previous);following=np.asarray(following)
    return np.concatenate((appearance,appearance-baseline,dynamic,appearance-previous,
                           following-appearance,np.asarray(short_following)-np.asarray(short_previous)),axis=1)

class SequenceClassifier(TypedClassifier):
    def category_probabilities(self,x):
        return self.estimator.predict_proba(sequence_features(x))
    def events(self,x,windows,threshold):
        events=super().events(x,windows,threshold)
        for e in events:
            # Covers the features actually used, unlike a purely decorative video clip.
            e['sequence_start']=max(0.,e['start']-12.)
            e['sequence_end']=min(windows[-1]['end'],e['end']+12.)
            e['description'] += ' Оценка учитывает изменения до и после фрагмента (до 12 секунд с каждой стороны). Отдельные этапы операции автоматически не подтверждены.'
        return events

def fit_sequence(training,c=.01):
    features=[sequence_features(r['x']) for r in training]
    selected=[np.flatnonzero(r['eligible']) for r in training]
    for iteration in range(3):
        x=np.concatenate([z[idx] for z,idx in zip(features,selected)])
        y=np.concatenate([np.repeat(r['label'],len(idx)) for r,idx in zip(training,selected)])
        counts={label:sum(r['label']==label for r in training) for label in set(y)}
        weight=np.concatenate([np.full(len(idx),1/(len(idx)*np.sqrt(counts[r['label']]))) for r,idx in zip(training,selected)])
        weight*=len(weight)/weight.sum()
        estimator=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1500,random_state=42))
        estimator.fit(x,y,logisticregression__sample_weight=weight)
        for i,r in enumerate(training):
            if not r['weak']: continue
            p=estimator.predict_proba(features[i])[:,list(estimator.classes_).index(r['label'])]
            # Select a contiguous 6-second region, not disconnected peak frames.
            width=min(3,len(p));start=int(np.convolve(p,np.ones(width)/width,'valid').argmax())
            selected[i]=np.arange(start,start+width)
    return SequenceClassifier(estimator)