"""Optional verified event-to-source bridge. Never infer a type from a binary score."""
from copy import deepcopy
from pathlib import Path

from ml.dataset import sha256
from ml.rules import GENERIC_TYPE, RuleEngine


def attach_verified_sources(result, settings):
    result = deepcopy(result)
    rules = RuleEngine(settings.rules)
    model_hash = None
    for event in result.get('violations', []):
        # Only administrator-approved bindings can add normative references.
        event.pop('rule_references', None)
        event_type = event.get('event_type')
        if not event_type or event.get('type') == GENERIC_TYPE:
            continue
        candidates = rules.sources_for(event_type)
        if not candidates:
            continue
        model_hash = model_hash or sha256(settings.model)
        references = []
        for rule in candidates:
            if rule.get('detector_model_sha256') != model_hash or not rule.get('validation_report'):
                continue
            report = Path(rule['validation_report'])
            if not report.is_absolute():
                report = settings.rules.parent / report
            if not report.is_file():
                continue
            references.append(dict(source_title=rule['source_title'], source_url=rule['source_url'],
                                   section=rule['section'], verified_at=rule['verified_at']))
        if references:
            event['rule_references'] = references
            sources = '; '.join(f'{r["source_title"]}, {r["section"]}: {r["source_url"]}' for r in references)
            event['description'] += f' Основание для ручной проверки: {sources}.'
    return result
