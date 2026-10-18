"""Optional verified event-to-source bridge. Never infer a type from a binary score."""
from copy import deepcopy
from pathlib import Path

from ml.dataset import sha256
from ml.rules import GENERIC_TYPE, RuleEngine


def attach_verified_sources(result, settings):
    result = deepcopy(result)
    rules = RuleEngine(settings.rules)
    model_hash = None
    for event in result.get('violations', []) + result.get('review_candidates', []):
        # Only administrator-approved bindings can add normative references.
        event.pop('rule_references', None)
        event_type = event.get('event_type')
        if not event_type or event.get('type') == GENERIC_TYPE:
            continue
        candidates = rules.sources_for(event_type)
        if not candidates:
            continue
        # Normative context does not certify detector accuracy.
        model_hash = model_hash or (sha256(settings.model) if settings.model.is_file() else None)
        references = []
        for rule in candidates:
            context = rule.get('reference_role') == 'normative_context'
            if context:
                if not rule.get('applicability') or not rule.get('source_audit'):
                    continue
                audit = settings.rules.parent / rule['source_audit']
                if not audit.is_file() or sha256(audit) != rule.get('source_audit_sha256'):
                    continue
            elif rule.get('detector_model_sha256') != model_hash or not rule.get('validation_report'):
                continue
            report = Path(rule['source_audit'] if context else rule['validation_report'])
            if not report.is_absolute():
                report = settings.rules.parent / report
            if not report.is_file():
                continue
            references.append(dict(source_title=rule['source_title'], source_url=rule['source_url'],
                                   section=rule['section'], verified_at=rule['verified_at'],
                                   applicability=rule.get('applicability'), reference_role=rule.get('reference_role', 'validated_binding')))
        if references:
            event['rule_references'] = references
            sources = '; '.join(f'{r["source_title"]}, {r["section"]}: {r["source_url"]}' for r in references)
            notes = ' '.join(dict.fromkeys(r['applicability'] for r in references if r.get('applicability')))
            event['description'] += f' Нормативный контекст для ручной проверки: {sources}. {notes}'
    return result
