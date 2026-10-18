"""Learn operation states separately; use state changes as transferable evidence."""
import numpy as np
from scipy.special import expit
from .generalized import features,GeneralizedClassifier

def state_features(x,heads):
    streams=[]
    for head in heads:
        p=head.predict_proba(x)[:,list(head.classes_).index(1)]
        past=np.array([p[max(0,i-3):i+1].mean() for i in range(len(p))])
        future=np.array([p[i:min(len(p),i+4)].mean() for i in range(len(p))])
        streams.extend([p,p-past,future-p,np.repeat(p.mean(),len(p))])
    return np.stack(streams,axis=1)

class ActionTransferClassifier(GeneralizedClassifier):
    def __init__(self,detector,heads,typed_head):
        self.detector=detector;self.heads=heads;self.typed_head=typed_head
    def predict_proba(self,x):
        p=expit(self.detector.decision_function(state_features(x,self.heads)))
        return np.column_stack([1-p,p])
