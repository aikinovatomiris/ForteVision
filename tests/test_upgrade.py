from ml.workflow import check_workflow
from tests.test_workflow import obs
from app.rule_links import attach_verified_sources
from app.config import Settings

def test_closure_is_not_locking():
    findings,checks=check_workflow(obs((0,'storage_closed'),(3,'cashier_leaves')))
    assert not findings
    assert next(c for c in checks if c['rule_id']=='storage_lock')['state']=='unknown'

def test_lock_observed_separately():
    _,checks=check_workflow(obs((0,'storage_closed'),(1,'storage_locked'),(3,'cashier_leaves')))
    assert next(c for c in checks if c['rule_id']=='storage_lock')['state']=='observed'

def test_explicit_unlocked_after_exit():
    findings,_=check_workflow(obs((0,'cashier_leaves'),(2,'storage_unlocked')))
    assert findings[0]['event_type']=='storage_left_unlocked'

def test_ambiguous_paper_is_not_document():
    findings,checks=check_workflow(obs((0,'document_candidate'),(1,'cash_received')))
    assert not findings
    assert next(c for c in checks if c['rule_id']=='identity_document')['state']=='unknown'

def test_runtime_legal_context_does_not_claim_validated_detector():
    result=attach_verified_sources({'violations':[dict(type='Документ',event_type='document_handover_not_observed',description='Гипотеза')]},Settings())
    ref=result['violations'][0]['rule_references'][0]
    assert ref['reference_role']=='normative_context'
    assert 'цифров' in ref['applicability']

def test_unknown_type_gets_no_legal_claim():
    result=attach_verified_sources({'violations':[dict(type='Неясно',description='Гипотеза')]},Settings())
    assert 'rule_references' not in result['violations'][0]
