from ml.workflow import check_workflow

def obs(*items):
    return [dict(timestamp_sec=t, action=a, frame_index=i, detail=a) for i,(t,a) in enumerate(items)]

def test_missing_action_is_unknown_not_violation():
    findings,checks = check_workflow(obs((0,'customer_arrives'),(5,'cash_received')))
    assert findings == []
    assert next(c for c in checks if c['rule_id']=='identity_document')['state']=='unknown'

def test_cash_after_leaving_differs_from_cash_before_leaving():
    before,_ = check_workflow(obs((0,'cash_on_counter'),(5,'cashier_leaves')))
    after,_ = check_workflow(obs((0,'cashier_leaves'),(5,'cash_on_counter')))
    assert before == []
    assert [f['event_type'] for f in after] == ['cash_left_unattended']

def test_customer_change_needs_cash_on_both_sides():
    events=obs((0,'cash_on_counter'),(2,'customer_leaves'),(5,'customer_arrives'),(7,'cash_on_counter'))
    findings,_=check_workflow(events)
    assert len(findings)==1
    assert check_workflow(events[1:])[0]==[]
    assert check_workflow(events+obs((4,'cash_into_storage')))[0]==[]

def test_closed_safe_and_stored_cash_no_violation():
    assert check_workflow(obs((0,'cash_into_storage'),(2,'storage_closed'),(4,'cashier_leaves')))[0]==[]

def test_open_safe_requires_departure():
    assert check_workflow(obs((4,'storage_open')))[0]==[]
    findings,_=check_workflow(obs((2,'cashier_leaves'),(4,'storage_open')))
    assert findings[0]['event_type']=='storage_left_open'


def test_mixing_needs_customer_transition_not_first_arrival():
    assert check_workflow(obs((0,'customer_arrives'),(5,'cash_added_to_existing_stack')))[0]==[]
    findings,_=check_workflow(obs((0,'customer_leaves'),(3,'customer_arrives'),(5,'cash_added_to_existing_stack')))
    assert findings[0]['event_type']=='prior_deposit_outside_storage'
