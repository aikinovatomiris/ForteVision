import numpy as np
import pytest
from ml.generalized import features,GeneralizedClassifier
from ml.multiscale import crops

def test_temporal_features_reject_nonfinite_input():
    bad=np.zeros((3,1024));bad[0,0]=np.nan
    with pytest.raises(ValueError):features(bad,'relative')

def test_relative_features_remove_unchanged_scene():
    x=np.tile(np.arange(1,513,dtype=float),(6,2))
    assert np.allclose(features(x,'relative'),0)

def test_detail_crops_cover_all_frame_pixels():
    frame=np.arange(80*50*3).reshape(80,50,3)
    detail=list(crops(frame));assert len(detail)==5
    values=set(np.concatenate([p[:,:,0].ravel() for p in detail[1:]]))
    assert values==set(frame[:,:,0].ravel())

class Detector:
    def decision_function(self,x):return np.repeat(3.,len(x))
class Estimator:
    classes_=np.array(['normal','door_left_open_after_entry'])
class Typed:
    estimator=Estimator()
    def category_probabilities(self,x):return np.tile([.99,.01],(len(x),1))

def test_unknown_type_does_not_suppress_binary_detection():
    model=GeneralizedClassifier(Detector(),'appearance',Typed())
    x=np.ones((3,1024));windows=[dict(start=i*2.,end=i*2.+4,timestamp=i*2.+2) for i in range(3)]
    events=model.events(x,windows,.7)
    assert events and all('event_type' not in e for e in events)
