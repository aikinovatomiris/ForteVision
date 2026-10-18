from ml.workflow import check_workflow
from tests.test_workflow import obs

def test_key_requires_visible_key_after_departure():
    assert check_workflow(obs((0,'key_in_storage_lock')))[0] == []
    assert check_workflow(obs((0,'key_in_storage_lock'),(5,'cashier_leaves')))[0] == []
    findings,_=check_workflow(obs((0,'cashier_leaves'),(2,'key_in_storage_lock')))
    assert [f['event_type'] for f in findings] == ['key_left_in_lock']

def test_key_does_not_mean_unlocked_door():
    findings,_=check_workflow(obs((0,'storage_locked'),(1,'cashier_leaves'),(2,'key_in_storage_lock')))
    assert [f['event_type'] for f in findings] == ['key_left_in_lock']
