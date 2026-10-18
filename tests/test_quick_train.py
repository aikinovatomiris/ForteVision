import joblib
import numpy as np
from ml.quick_train import RidgeClassifier

def test_ridge_roundtrip_and_context_offset_invariance(tmp_path):
    rng=np.random.default_rng(42)
    x=np.r_[rng.normal(-1,.1,(8,12)),rng.normal(1,.1,(8,12))]
    y=np.r_[np.zeros(8),np.ones(8)]
    model=RidgeClassifier().fit(x-np.median(x,axis=0),y,np.ones(16),10)
    model.context_center_=True
    before=model.predict_proba(x)
    assert np.array_equal(before[:,1]>=.5,y)
    assert np.allclose(before,model.predict_proba(x+np.arange(12)*10))
    path=tmp_path/'model.joblib'
    joblib.dump(model,path)
    assert np.allclose(before,joblib.load(path).predict_proba(x))
    assert np.allclose(before.sum(axis=1),1)

def test_full_frame_keeps_edge_content():
    from ml.detection import ResNetExtractor
    from torchvision.models import ResNet18_Weights
    extractor=object.__new__(ResNetExtractor)
    extractor.image_mode='full_frame'
    extractor.transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
    frame=np.zeros((160,320,3),dtype=np.uint8)
    frame[:,:20,2]=255
    tensor=extractor.preprocess(frame)
    assert tuple(tensor.shape)==(3,320,320)
    # Red at the extreme left survives aspect-preserving padding.
    red=tensor[0,160,0].item()*.229+.485
    assert red>.99