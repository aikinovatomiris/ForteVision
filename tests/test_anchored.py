import numpy as np
import pytest
from ml.anchored import AnchoredClassifier, targets, features
from scripts.train_anchored import build_records, measure, group, release_gate


def record(label=1):
    return dict(row={'id':'x-1'},binary_label=label,x=np.zeros((4,1024)),
        windows=[dict(start=t-1,end=t+1,timestamp=t) for t in (2,4,6,12)],
        events=[dict(time=4,type='cash_left_unattended')] if label else [])


def test_unannotated_positive_windows_are_unknown():
    y,typed=targets(record())
    assert y.tolist()==[1,1,1,-1]
    assert typed['cash_left_unattended'].tolist()==[True,True,True,False]
    assert targets(record(0))[0].tolist()==[0,0,0,0]


def test_duplicate_conflicting_workbook_labels_fail():
    manifest={'rows':[{'id':'x-4','aliases':['Х-47.mov']}]}
    source={'x-4':{'binary_label':1},'x-47':{'binary_label':0}}
    with pytest.raises(ValueError,match='conflicting'):
        build_records(manifest,source,{'events':{},'recording_groups':[]})


def test_related_recordings_share_group():
    assert group('x-4',[['x-4','x-5']])==group('x-5',[['x-4','x-5']])
    assert group('x-6',[['x-4','x-5']])!='x-4|x-5'


def test_localization_matches_events_one_to_one_and_counts_misses():
    r=record();r['events'] += [dict(time=6,type='cash_left_unattended')]
    report=measure([r],[[dict(peak=5,event_type='cash_left_unattended')]],.5)
    assert report['annotated_events']==2
    assert report['matched_within_5_seconds']==1
    assert report['correct_type_and_time']==1
    missed=measure([r],[[]],.5)
    assert missed['fn']==1 and missed['matched_mae_seconds'] is None


class Fixed:
    def __init__(self,p): self.p=np.array(p)
    def predict_proba(self,x): return np.column_stack([1-self.p,self.p])


def test_multiple_types_and_unknown_are_preserved():
    r=record()
    m=AnchoredClassifier(Fixed([.9]*4),{
        'cash_left_unattended':Fixed([.9,.9,0,0]),
        'storage_left_open':Fixed([0,.9,.9,0])},'appearance')
    events=m.events(r['x'],r['windows'],.5)
    assert {e.get('event_type') for e in events}=={'cash_left_unattended','storage_left_open',None}
    assert all(e['peak'] in [2,4,6,12] for e in events)


def test_invalid_features_fail():
    with pytest.raises(ValueError): features(np.zeros((1,512)),'appearance')
    with pytest.raises(ValueError): features(np.zeros((1,1024)),'typo')


def test_relative_features_have_no_absolute_appearance_block():
    x=np.random.default_rng(42).normal(size=(8,1024))
    from ml.sequence import sequence_features
    assert np.array_equal(features(x,'relative'),sequence_features(x)[:,512:])


def test_all_positive_detector_cannot_pass_release_gate():
    assert not release_gate(dict(precision=.5,recall=1.,tn=0,
        correct_type_and_time=4,annotated_events=4))['allowed']
