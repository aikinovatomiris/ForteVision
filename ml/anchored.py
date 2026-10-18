"""Timestamp-supervised visual hypotheses. No filenames/text enter prediction."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml.sequence import sequence_features
from ml.temporal import merge_events

CATALOG = {
    'manual_recount_not_observed': ('Ручной пересчёт не наблюдается', 'После машинного пересчёта возможен переход к следующему действию без наблюдаемого ручного пересчёта.'),
    'document_handover_not_observed': ('Передача документа не наблюдается', 'Возможен приём денег без наблюдаемой передачи документа. Идентификация вне кадра не проверяется.'),
    'prior_deposit_outside_storage': ('Деньги предыдущего клиента не убраны', 'Возможно начало обслуживания следующего клиента, пока деньги предыдущего остаются на рабочей поверхности.'),
    'computer_left_on': ('Компьютер оставлен после ухода', 'Возможно, кассир покинул рабочее место, оставив компьютер включённым. Состояние блокировки экрана не установлено.'),
    'deposit_stored_without_observed_count': ('Деньги убраны без наблюдаемого пересчёта', 'Возможное помещение принятых денег в хранилище без наблюдаемого пересчёта в предшествующем фрагменте.'),
    'cash_left_unattended': ('Деньги оставлены без присмотра', 'Возможен уход кассира при оставшихся на рабочем столе деньгах или пакете с деньгами.'),
    'overlapping_customer_service': ('Одновременное обслуживание клиентов', 'Возможно начало обслуживания второго клиента до завершения обслуживания первого.'),
    'storage_left_open': ('Хранилище предположительно не закрыто', 'Возможен переход к следующему действию или уход без наблюдаемого закрытия хранилища. Положение дверцы не доказывает состояние замка.'),
    'light_left_on': ('Свет оставлен после ухода', 'Возможен уход кассира при включённом освещении. Сам по себе этот признак не устанавливает нарушение нормы.'),
    'door_left_open': ('Дверь оставлена открытой', 'Возможно, после прохода дверь рабочего помещения осталась открытой.'),
    'authenticity_check_not_observed': ('Проверка банкнот не наблюдается', 'Возможен переход от счётной машинки к складыванию денег без наблюдаемого использования отдельного прибора проверки. Подлинность банкнот не определяется.'),
    'food_drink_at_workstation': ('Еда или напитки рядом с деньгами', 'Возможно размещение еды или употребление напитка на рабочем столе рядом с наличными.'),
}


def features(x, mode):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 1024 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite 1024-dimensional window features')
    if mode not in ('appearance', 'sequence', 'relative'):
        raise ValueError('Unknown feature mode')
    if mode == 'relative':
        # Remove absolute appearance to reduce room/clothing shortcuts.
        return sequence_features(x)[:, 512:]
    return sequence_features(x) if mode == 'sequence' else x


class AnchoredClassifier:
    classes_ = np.array([0, 1])
    n_features_in_ = 1024

    def __init__(self, binary, types, mode, type_threshold=.6):
        self.binary, self.types, self.mode = binary, types, mode
        self.type_threshold = type_threshold

    def predict_proba(self, x):
        return self.binary.predict_proba(features(x, self.mode))

    def events(self, x, windows, threshold):
        z = features(x, self.mode)
        scores = self.binary.predict_proba(z)[:, 1]
        result = []
        covered = np.zeros(len(x), dtype=bool)
        for name, estimator in self.types.items():
            p = estimator.predict_proba(z)[:, 1]
            active = (scores >= threshold) & (p >= self.type_threshold)
            covered |= active
            # Distinct types may coexist; no multiclass argmax suppresses an event.
            for event in merge_events(windows, np.where(active, scores*p, -1), 0):
                title, description = CATALOG[name]
                event.update(type=title, event_type=name,
                    description='Гипотеза модели: '+description+' Требует проверки последовательности кадров; юридический вывод не установлен.')
                result.append(event)
        for event in merge_events(windows, np.where(~covered, scores, -1), threshold):
            event.update(type='Признаки возможного нарушения',
                description='Модель выделила подозрительный фрагмент, но не определила тип действия достаточно уверенно. Требуется просмотр видео.')
            result.append(event)
        if self.mode != 'appearance':
            for event in result:
                event['sequence_start'] = max(0., event['start']-12)
                event['sequence_end'] = min(windows[-1]['end'], event['end']+12)
        return sorted(result, key=lambda e: e['peak'])


def targets(record):
    """Unknown positive-video regions are excluded, never silently normal."""
    n = len(record['x'])
    y = np.full(n, -1, dtype=int)
    if record['binary_label'] == 0:
        y[:] = 0
    typed = {}
    for event in record['events']:
        mask = np.array([abs(w['timestamp']-event['time']) <= 2.1 for w in record['windows']])
        if not mask.any():
            raise ValueError('Annotation has no sampled window')
        y[mask] = 1
        typed[event['type']] = typed.get(event['type'], np.zeros(n, bool)) | mask
    return y, typed


def fit(records, mode='appearance', c=.01, weak_negative_weight=0.):
    rows = []
    for r in records:
        y, typed = targets(r)
        strength = np.ones(len(y))
        if weak_negative_weight and r['events']:
            # After an event an unsafe state may persist: never call that normal.
            distant = np.array([w['end'] < min(e['time'] for e in r['events'])-10
                                for w in r['windows']]) & (y < 0)
            y[distant] = 0
            strength[distant] = weak_negative_weight
        known = y >= 0
        if known.any():
            rows.append((features(r['x'], mode)[known], y[known],
                         {k: v[known] for k, v in typed.items()}, strength[known]))
    x = np.concatenate([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    if set(y) != {0, 1}:
        raise ValueError('Both positive and normal training windows required')
    weight = np.concatenate([r[3]/r[3].sum() for r in rows])
    weight *= len(weight)/weight.sum()

    def estimator(labels):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced',
            max_iter=2000, random_state=42, solver='liblinear'))
        model.fit(x, labels, logisticregression__sample_weight=weight)
        return model

    binary = estimator(y)
    types = {}
    for name in sorted({k for r in rows for k in r[2]}):
        labels = np.concatenate([r[2].get(name, np.zeros(len(r[1]), bool)) for r in rows]).astype(int)
        if len(set(labels)) == 2:
            types[name] = estimator(labels)
    return AnchoredClassifier(binary, types, mode)
