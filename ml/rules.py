import json
from pathlib import Path

from .dataset import ForteError

GENERIC_TYPE = 'Признаки возможного нарушения'


class RuleEngine:
    def __init__(self, path=Path('configs/rules.json')):
        self.rules = json.loads(Path(path).read_text(encoding='utf-8'))['rules']
        for rule in self.rules:
            if rule.get('event_type') == GENERIC_TYPE:
                raise ForteError('Общий бинарный score нельзя связывать с конкретной нормой')
            if not all(rule.get(k) for k in ('event_type', 'source_title', 'source_url', 'section', 'verified_by', 'verified_at')):
                raise ForteError('Нормативный источник не проверен: неполная запись')

    def sources_for(self, event_type):
        if event_type is None or event_type == GENERIC_TYPE:
            return []
        return [rule for rule in self.rules if rule['event_type'] == event_type]
