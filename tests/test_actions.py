import numpy as np
from ml.action_model import ActionAwareClassifier

class Typed:
    def events(self,x,w,t):
        return [dict(start=4.,end=8.,peak=6.,event_type='storage_left_open',type='safe',description='')]
    def predict_proba(self,x):return np.tile([.1,.9],(len(x),1))

class Presence:
    classes_=np.array([0,1])
    def __init__(self,p):self.p=np.array(p)
    def predict_proba(self,x):return np.column_stack([1-self.p,self.p])

def run(p):
    windows=[dict(start=i*2.,end=i*2.+4.,timestamp=i*2.+2.) for i in range(len(p))]
    return ActionAwareClassifier(Typed(),Presence(p)).events(np.zeros((len(p),1024)),windows,.55)

def test_seated_cashier_cannot_be_reported_as_departed():
    events=run([.95,.97,.99,.96])
    assert len(events)==1 and 'event_type' not in events[0]
    assert 'не подтвердила' in events[0]['description']

def test_empty_scene_without_prior_employee_is_not_departure():
    assert 'event_type' not in run([.02,.02,.02,.02])[0]

def test_present_then_absent_supports_departure_hypothesis():
    assert run([.99,.98,.1,.05])[0]['event_type']=='storage_left_open'

def test_uncertain_presence_does_not_prove_departure():
    assert 'event_type' not in run([.51,.49,.51,.49])[0]
