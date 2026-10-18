"""Local-only chronological VLM observations followed by deterministic workflow rules."""
import base64,json,math
from pathlib import Path
from time import perf_counter
import cv2,numpy as np,requests
from .dataset import ForteError
from .preprocessing import probe,read_at_times,check_deadline
from .workflow import ACTIONS,check_workflow
ROOT=Path(__file__).resolve().parents[1]
URL='http://127.0.0.1:8091'

# Explicit structural output; strings from video/model are data, never executed.
SCHEMA={'type':'object','properties':{'observations':{'type':'array','maxItems':20,'items':{'type':'object','properties':{'frame_index':{'type':'integer','minimum':0},'action':{'type':'string','enum':list(ACTIONS)},'detail':{'type':'string'}},'required':['frame_index','action','detail'],'additionalProperties':False}},'limitations':{'type':'array','items':{'type':'string'}}},'required':['observations','limitations'],'additionalProperties':False}

def local_session():
    session=requests.Session();session.trust_env=False
    return session

def ready():
    try:
        with local_session() as s:return s.get(URL+'/health',timeout=2,allow_redirects=False).status_code==200
    except requests.RequestException:return False

def observe(frames,deadline):
    prompt=(ROOT/'configs/workflow_prompt.txt').read_text(encoding='utf-8')
    content=[{'type':'text','text':'Chronological frames follow. Only report what is visible; there may be no violation.'}]
    for i,(timestamp,frame) in enumerate(frames):
        scale=min(1.,896/max(frame.shape[:2]))
        if scale<1: frame=cv2.resize(frame,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        ok,data=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
        if not ok:raise ForteError('JPEG encoding failed')
        content.append({'type':'text','text':f'frame_index={i}; timestamp={timestamp:.3f} seconds'})
        content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(data).decode('ascii')}})
    payload=dict(messages=[{'role':'system','content':prompt},{'role':'user','content':content}],temperature=0.1,top_p=0.8,top_k=20,repeat_penalty=1.15,seed=42,max_tokens=1100,
                 response_format={'type':'json_schema','json_schema':{'name':'observations','strict':True,'schema':SCHEMA}},
                 chat_template_kwargs={'enable_thinking':False})
    check_deadline(deadline)
    with local_session() as session:
        response=session.post(URL+'/v1/chat/completions',json=payload,timeout=(5,max(1,deadline-perf_counter())),allow_redirects=False)
    response.raise_for_status()
    body=response.json();choice=body['choices'][0]
    if choice.get('finish_reason')=='length':raise ForteError('VLM response truncated; no verdict produced')
    parsed=json.loads(choice['message']['content'])
    observations=[];seen=set()
    for item in parsed['observations']:
        index=item['frame_index'];action=item['action']
        if type(index) is not int or not 0<=index<len(frames) or action not in ACTIONS:raise ForteError('Invalid visual observation')
        if (index,action) in seen:continue
        seen.add((index,action))
        observations.append(dict(frame_index=index,timestamp_sec=frames[index][0],action=action,detail=str(item['detail'])[:500]))
    return observations,parsed.get('limitations',[]),body.get('timings',{})

def analyze(video,budget_seconds=285):
    start=perf_counter();deadline=start+budget_seconds
    if not ready():raise ForteError('Локальная видеоязыковая модель не запущена на 127.0.0.1:8091')
    info=probe(video)
    duration=info['duration_seconds']
    if duration>3600:raise ForteError('Видео превышает лимит длительности')
    # Whole-video overview; no classification threshold gates access to the frames.
    count=min(24,max(8,math.ceil(duration)+1))
    targets=np.linspace(0,max(0,duration-1/info['fps']),count)
    frames=read_at_times(video,targets,deadline)
    observations=[];limitations=[];timings=[]
    for offset in range(0,len(frames)-1,3):
        chunk=frames[offset:offset+4]
        items,notes,timing=observe(chunk,deadline)
        for item in items:
            item['frame_index']+=offset
            if not any(o['frame_index']==item['frame_index'] and o['action']==item['action'] for o in observations):
                observations.append(item)
        limitations.extend(notes);timings.append(timing)
    findings,checks=check_workflow(observations)
    violations=[]
    for finding in findings:
        evidence=[]
        for observation in finding['evidence'][:3]:
            timestamp,frame=frames[observation['frame_index']]
            scale=min(1.,960/max(frame.shape[:2]))
            if scale<1:frame=cv2.resize(frame,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
            ok,data=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
            if not ok:raise ForteError('JPEG encoding failed')
            if any(e['timestamp_sec']==timestamp for e in evidence):continue
            evidence.append(dict(timestamp_sec=timestamp,image_base64=base64.b64encode(data).decode('ascii')))
        timestamps=[e['timestamp_sec'] for e in evidence]
        violations.append(dict(type=finding['type'],event_type=finding['event_type'],timecode_sec=finding['peak'],
            description=finding['description']+' Это предположение по выборочным кадрам, не подтверждённое нарушение нормы.',
            evidence_frames=evidence,context_start_sec=max(0,min(timestamps)-5),context_end_sec=min(duration,max(timestamps)+5)))
    check_deadline(deadline)
    elapsed=perf_counter()-start
    return dict(verdict='violation' if violations else 'no_violation',violations=violations,
        elapsed_seconds=elapsed,within_five_minutes=elapsed<300,workflow_checks=checks,
        observations=observations,analysis_limitations=['Выборочные кадры: короткие действия могут быть пропущены. Отсутствие замеченного действия не доказывает его отсутствие.']+[str(t)[:500] for t in limitations],
        analysis_complete=False,analysis_backend='local_vlm_workflow',sampled_frames=len(frames),vlm_timings=timings)