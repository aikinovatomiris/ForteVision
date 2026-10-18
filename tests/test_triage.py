from ml.triage import route_events,add_review_candidates

def event(score=.9,kind='storage_left_open'):
    return dict(start=3.,end=7.,peak=5.,score=score,event_type=kind,type='hypothesis')

def test_unknown_never_becomes_alert_even_with_high_score():
    alerts,review=route_events([event(.999,None)])
    assert not alerts and len(review)==1

def test_unvalidated_category_abstains_without_losing_moment():
    original=event();alerts,review=route_events([original],validated_types=set())
    assert not alerts and review[0]['peak']==original['peak']
    assert review[0]['event_type']==original['event_type']
    assert 'review_reason' not in original

def test_routing_preserves_all_intervals_including_brief_events():
    events=[event(.9),dict(event(.2),start=9.,end=9.1,peak=9.05),event(.99,None)]
    a,r=route_events(events)
    assert sorted(e['peak'] for e in a+r)==sorted(e['peak'] for e in events)

def test_sensitive_candidate_cannot_promote_itself():
    a,r=route_events([dict(event(.99),review_only=True)])
    assert not a and len(r)==1

def test_same_peak_different_intervals_not_discarded():
    primary=event();extra=dict(event(),start=1.,end=10.)
    combined=add_review_candidates([primary],[extra])
    assert len(combined)==2 and combined[0]['start']==3.
