"""Action supervision separate from video-level violation categories."""
import numpy as np


class ActionAwareClassifier:
    classes_ = np.array([0, 1])
    n_features_in_ = 1024

    def __init__(self, typed, presence):
        self.typed = typed
        self.presence = presence

    def predict_proba(self, x):
        return self.typed.predict_proba(x)

    def events(self, x, windows, threshold):
        events = self.typed.events(x, windows, threshold)
        p = self.presence.predict_proba(x)
        present = p[:, list(self.presence.classes_).index(1)]
        for event in events:
            if event.get('event_type') != 'storage_left_open':
                continue
            # Require an earlier occupied workstation and later unoccupied
            # workstation. Customer departure is not employee departure.
            after = [i for i, w in enumerate(windows)
                     if event['start'] <= w['timestamp'] <= event['end'] + 4]
            before = [i for i, w in enumerate(windows)
                      if w['timestamp'] < event['peak']]
            supported = (before and after and max(present[before]) >= .65
                         and min(present[after]) <= .35)
            if not supported:
                event.pop('event_type', None)
                event['type'] = 'Последовательность ухода не подтверждена'
                event['description'] = ('Категорийная модель выделила фрагмент, но отдельная модель присутствия '
                    'кассира не подтвердила последовательность ухода. Оставление хранилища не установлено; '
                    'нужна ручная проверка. Неопределённость не означает отсутствие нарушения.')
        return events
