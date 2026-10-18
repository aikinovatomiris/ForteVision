import numpy as np
from ml.typed import TypedClassifier

class Estimator:
    classes_ = np.array(['cash_left_unattended','normal','storage_left_open'])
    def __init__(self,p): self.p=np.array(p)
    def predict_proba(self,x): return self.p

def test_actions_not_merged_and_normal_suppressed():
    m=TypedClassifier(Estimator([[.9,.05,.05],[.05,.05,.9],[.05,.9,.05]]))
    w=[dict(start=i*2,end=i*2+4,timestamp=i*2+1) for i in range(3)]
    events=m.events(np.zeros((3,1024)),w,.55)
    assert [e['event_type'] for e in events]==['cash_left_unattended','storage_left_open']
    assert all('гипотеза' in e['description'] for e in events)
    assert np.allclose(m.predict_proba(None).sum(axis=1),1)

def test_uncertain_type_is_not_invented():
    m=TypedClassifier(Estimator([[.4,.2,.4]]))
    events=m.events(None,[dict(start=0,end=4,timestamp=2)],.55)
    assert len(events)==1
    assert 'event_type' not in events[0]
    assert events[0]['type']=='Тип события не определён'