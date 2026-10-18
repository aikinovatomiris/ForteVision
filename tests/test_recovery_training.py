import numpy as np
import pytest

from ml.recovery_training import reconcile, evaluate


def row(ident, split='train', label=1, aliases=None):
    return dict(id=ident, split=split, label=label, timestamps=[57., 68.] if label else [],
                aliases=aliases or [], file=ident+'.mov', sha256=ident)


def test_explicit_time_override_preserves_test_and_unknown_labels():
    manifest = dict(rows=[row('x-4', aliases=['Х-47.mov']), row('x-16', 'test', 0),
                          row('x-17'), row('x-29', 'validation', 0)])
    annotations = {i: dict(binary_label=y, description='source') for i,y in
                   [('x-4',1),('x-47',1),('x-16',0),('x-17',1),('x-29',None)]}
    mapping = dict(types={'x-4':'prior_deposit_outside_storage','x-47':'prior_deposit_outside_storage',
                          'x-17':'storage_left_open'},
                   timestamp_overrides={'x-4':dict(timestamps=[57.],source='user')})
    records, _ = reconcile(manifest, annotations, mapping)
    assert records[0]['timestamps'] == [57.]
    assert records[1]['row']['split'] == 'test'
    assert records[2]['excluded_from_fitting']
    assert records[3]['binary_label'] is None and records[3]['label'] is None
    assert manifest['rows'][0]['timestamps'] == [57.,68.]


def test_duplicate_alias_conflict_fails_closed():
    manifest = dict(rows=[row('x-4', aliases=['X-47.mov'])])
    with pytest.raises(ValueError, match='Conflicting workbook'):
        reconcile(manifest, {'x-4':{'binary_label':1},'x-47':{'binary_label':0}}, {'types':{}})


def test_new_negative_does_not_keep_old_positive_timestamp():
    records, conflicts = reconcile(dict(rows=[row('x-24','validation')]),
                                    {'x-24':{'binary_label':0}}, {'types':{}})
    assert records[0]['label']=='normal' and records[0]['timestamps']==[]
    assert conflicts[0]['workbook']==0


def test_related_recordings_stay_together_in_diagnostic():
    from scripts.evaluate_recovery_groups import recording_group
    assert recording_group({'row':row('x-12')}) == recording_group({'row':row('x-24')})
    assert recording_group({'row':row('x-12')}) != recording_group({'row':row('x-4')})


def test_no_predictions_are_counted_as_misses_not_localization_success():
    class Model:
        estimator = type('Estimator', (), {'classes_':np.array(['normal','prior_deposit_outside_storage'])})()
        def events(self, *args):return []
    record = dict(row=row('x-4'), binary_label=1, label='prior_deposit_outside_storage',
                  x=[], windows=[], timestamps=[57.],timestamp_source='user')
    report = evaluate(Model(), [record], .5)
    assert report['fn']==1 and report['conditional_nearest_peak_mae'] is None


def test_review_candidate_metadata_routes_typed_events_without_presence_head():
    from unittest.mock import patch
    from pathlib import Path
    from ml.pipeline import analyze
    class Classifier:
        def predict_proba(self, x):return np.array([[.1,.9]])
        def events(self,*args):
            return [dict(start=54.,end=58.,peak=56.,score=.9,event_type='prior_deposit_outside_storage',
                         type='Деньги предыдущего клиента',description='Гипотеза')]
    class Extractor:
        weights_sha256='fixture'
        def __init__(self,*args):pass
        def extract(self,*args):return np.zeros((1,1024)),[],dict(duration_seconds=73.)
    metadata=dict(sampling={},weights_sha256='fixture',threshold=.55,review_only=True)
    with patch('ml.pipeline.load_classifier',return_value=(Classifier(),metadata)), \
         patch('ml.pipeline.ResNetExtractor',Extractor), patch('ml.pipeline.evidence_frames',return_value=[]):
        result=analyze(Path('video.mov'),Path('candidate.joblib'),Path('weights.pth'))
    assert result['violations']==[] and result['assessment_status']=='needs_review'
    assert result['analysis_complete'] is False
    assert result['review_candidates'][0]['event_type']=='prior_deposit_outside_storage'
