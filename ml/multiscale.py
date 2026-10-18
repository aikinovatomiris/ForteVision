"""Camera-independent full image and overlapping detail crops, pooled without position IDs."""
import numpy as np
from .detection import ResNetExtractor
from .preprocessing import probe,iter_frames,pooled_windows,check_deadline

def crops(frame):
    h,w=frame.shape[:2]
    if min(h,w)<4:raise ValueError('Frame too small')
    yield frame
    for x0,y0,x1,y1 in ((0,0,.65,.65),(.35,0,1,.65),(0,.35,.65,1),(.35,.35,1,1)):
        yield frame[int(y0*h):int(y1*h),int(x0*w):int(x1*w)]

class MultiScaleExtractor(ResNetExtractor):
    def __init__(self,weights_path,device='cpu'):
        super().__init__(weights_path,device,'full_frame')
    def extract(self,path,config,deadline=None):
        info=probe(path);pending=[];times=[];all_times=[];vectors=[]
        frames_per_batch=max(1,config.batch_size//5)
        def flush():
            if not times:return
            z=self.embed(pending).reshape(len(times),5,512)
            vectors.extend(z.max(axis=1));all_times.extend(times)
            pending.clear();times.clear();check_deadline(deadline)
        for timestamp,frame in iter_frames(path,config,deadline):
            pending.extend(self.preprocess(c) for c in crops(frame));times.append(timestamp)
            if len(times)>=frames_per_batch:flush()
        flush()
        x,windows=pooled_windows(all_times,np.asarray(vectors),config,info['duration_seconds'])
        return x,windows,dict(**info,sampled_frames=len(all_times),crops_per_frame=5,timestamp_source='opencv_pos_msec')
