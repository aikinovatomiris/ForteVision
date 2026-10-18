import numpy as np
from ml.action_transfer import state_features

class Head:
    classes_=np.array([0,1])
    def predict_proba(self,x):
        p=x[:,0];return np.column_stack([1-p,p])

def test_states_encode_direction_of_transition():
    x=np.zeros((8,1024));x[:4,0]=.95;x[4:,0]=.05
    z=state_features(x,[Head()])
    assert z.shape==(8,4)
    assert z[4,1]<0 and z[2,2]<0
    assert np.allclose(z[:,3],.5)

def test_constant_state_has_no_transition():
    x=np.full((8,1024),.4)
    z=state_features(x,[Head(),Head()])
    assert np.allclose(z[:,[1,2,5,6]],0)
