"""Local, weakly supervised event categories from team annotations.
Video-level labels without timestamps do not claim precise action boundaries.
"""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .dataset import sha256, write_json, key
from .temporal import merge_events

CATALOG = {
'door_left_open_after_entry': ('Дверь оставлена открытой после входа кассира', 'Возможно, кассир вошла в рабочее помещение и не закрыла за собой дверь. Проверьте последовательность входа и положение двери после прохода; краткое открытие во время прохода само по себе не означает нарушение.'),
'key_left_in_lock': ('Ключ оставлен в замке сейфа после ухода', 'Возможно, кассир оставила ключ в замке сейфа и покинула рабочее место. Проверьте ключ на кадрах после ухода. Наличие ключа не означает, что замок не заперт.'),
'cash_left_unattended': ('Наличные оставлены без присмотра', 'Кассир покинул рабочее место, оставив наличные на столе.'),
'prior_deposit_outside_storage': ('Деньги предыдущего клиента не убраны', 'Следующая операция началась, пока ранее принятые деньги оставались на рабочей поверхности. Проверьте, не добавлены ли деньги следующего клиента к предыдущей стопке: это требует просмотра всей последовательности, а не одного кадра.'),
'storage_left_open': ('Хранилище оставлено открытым', 'Возможно, после ухода кассира хранилище осталось открытым. Проверьте дверцу и замок отдельно: закрытие дверцы не доказывает запирание ключом.'),
'document_handover_not_observed': ('Не наблюдается передача документов', 'Возможен пропуск предъявления документа при приёме наличных. Проверьте взаимодействие с клиентом целиком: банкнота или квитанция не являются удостоверением. Возможны цифровая идентификация и проверка до начала записи; модель их не устанавливает.'),
'recount_not_observed': ('Возможный пропуск пересчёта наличных', 'В операции не наблюдается ожидаемый этап пересчёта наличных. Способ и полноту пересчёта нужно проверить по записи.'),
'workstation_left_unsecured': ('Рабочее место оставлено незакрытым', 'После ухода кассира оборудование, записи или место хранения остались без наблюдаемого закрытия.'),
'overlapping_customer_service': ('Одновременное обслуживание клиентов', 'Обслуживание следующего клиента началось до завершения предыдущего.'),
'door_left_open': ('Дверь рабочего помещения оставлена открытой', 'После прохода кассира дверь рабочего помещения не была закрыта.'),
'workstation_left_unattended': ('Клиент у рабочего места без кассира', 'Клиент подошёл к рабочему месту в отсутствие кассира.'),
'personal_items_on_workstation': ('Личные вещи на рабочей поверхности', 'Кассир разместил личные вещи на рабочей поверхности для кассовых операций.'),
'separated_cash_remixed': ('Смешивание разделённых стопок купюр', 'После машинной обработки разделённые стопки купюр объединены. Подлинность купюр по этому признаку определить нельзя.'),
'food_drink_at_workstation': ('Еда или напитки рядом с наличными', 'Кассир ест или пьёт на рабочем месте рядом с наличными.')}

class TypedClassifier:
    classes_ = np.array([0, 1])
    n_features_in_ = 1024
    def __init__(self, estimator):
        self.estimator = estimator
    def category_probabilities(self, x):
        return self.estimator.predict_proba(x)
    def predict_proba(self, x):
        p = self.category_probabilities(x)
        normal = p[:, list(self.estimator.classes_).index('normal')]
        return np.column_stack((normal, 1-normal))
    def events(self, x, windows, threshold):
        p = self.category_probabilities(x)
        classes = list(self.estimator.classes_)
        winner = p.argmax(axis=1)
        result = []
        # Never merge different predicted actions into a single event.
        for col, name in enumerate(classes):
            if name == 'normal': continue
            score = np.where(winner == col, p[:, col], 0.)
            for event in merge_events(windows, score, threshold):
                title, description = CATALOG[name]
                event.update(type=title, event_type=name,
                    description='Предполагаемое действие: '+description+' Это гипотеза модели по визуальному сходству; проверьте последовательность кадров. Юридический вывод не установлен.')
                result.append(event)
        binary = 1-p[:, classes.index('normal')]
        # Preserve uncertain detections rather than inventing a specific type.
        unsure = np.where(p.max(axis=1) < threshold, binary, 0.)
        for event in merge_events(windows, unsure, threshold):
            event.update(type='Тип события не определён', description='Модель выделила необычный фрагмент, но не смогла уверенно выбрать действие. Просмотрите запись около указанного времени.')
            result.append(event)
        return sorted(result, key=lambda e:e['peak'])

def records(root):
    manifest=json.loads((root/'artifacts/quick_manifest.json').read_text(encoding='utf-8'))
    annotations=json.loads((root/'data/raw/team_annotations.json').read_text(encoding='utf-8'))
    negative={f'x-{i}' for i in annotations['team_negative_ids']}
    output=[]
    for row in manifest['rows']:
        ids=[row['id']]+[key(n) for n in row.get('aliases',[])]
        types={annotations['events'][i][0] for i in ids if i in annotations['events']}
        if types and any(i in negative for i in ids): raise ValueError('Conflicting team duplicate group')
        if len(types)>1: raise ValueError('Multiple types require timestamp annotations')
        label=next(iter(types)) if types else 'normal' if any(i in negative for i in ids) else None
        if label is None: continue
        cache=root/'artifacts/quick_cache_full_frame'/f"{row['sha256']}.joblib"
        data=joblib.load(cache)
        baseline=json.loads((root/'models/fullframe_classifier.json').read_text())
        if data['weights_sha256']!=baseline['weights_sha256'] or data['sampling']!=baseline['sampling']:
            raise ValueError('Cached features do not match extractor')
        # Original event times are usable only when the binary annotation agrees.
        times=row['timestamps'] if row['label']==1 and label!='normal' else []
        eligible=np.ones(len(data['x']),dtype=bool)
        if times:
            eligible=np.array([any(w['start']<=t+1 and w['end']>t-1 for t in times) for w in data['windows']])
        output.append(dict(row=row,label=label,x=data['x'],windows=data['windows'],eligible=eligible,weak=label!='normal' and not times))
    return output

def fit(training):
    selected=[np.flatnonzero(r['eligible']) for r in training]
    for iteration in range(4):
        x=np.concatenate([r['x'][idx] for r,idx in zip(training,selected)])
        y=np.concatenate([np.repeat(r['label'],len(idx)) for r,idx in zip(training,selected)])
        weight=np.concatenate([np.full(len(idx),1/len(idx)) for idx in selected])
        # Equal total contribution for each video, no frame-wise data splitting.
        weight*=len(weight)/weight.sum()
        estimator=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=2000))
        estimator.fit(x,y,logisticregression__sample_weight=weight)
        for i,r in enumerate(training):
            if not r['weak']: continue
            score=estimator.predict_proba(r['x'])[:,list(estimator.classes_).index(r['label'])]
            selected[i]=np.sort(np.argsort(score)[-min(3,len(score)):])
    return TypedClassifier(estimator)

def main():
    root=Path(__file__).resolve().parents[1]
    model_path=root/'models/typed_classifier.joblib'
    if model_path.exists(): raise ValueError('Refusing to overwrite typed model')
    rows=records(root)
    # Final development training uses prior train+validation; original test stays out.
    training=[r for r in rows if r['row']['split']!='test']
    test=[r for r in rows if r['row']['split']=='test']
    model=fit(training)
    threshold=.55  # fixed before evaluation; uncalibrated on this small dataset
    details=[]
    for r in test:
        events=model.events(r['x'],r['windows'],threshold)
        details.append(dict(id=r['row']['id'],expected=r['label'],predicted=sorted({e.get('event_type','unknown') for e in events}),
                            positive=bool(events),label_seen_in_training=r['label'] in model.estimator.classes_))
    tp=sum(r['positive'] and r['expected']!='normal' for r in details)
    fp=sum(r['positive'] and r['expected']=='normal' for r in details)
    fn=sum(not r['positive'] and r['expected']!='normal' for r in details)
    tn=sum(not r['positive'] and r['expected']=='normal' for r in details)
    report=dict(confusion_matrix=[[tn,fp],[fn,tp]],f1=2*tp/max(1,2*tp+fp+fn),videos=details,
                evaluation_note='Previously examined development holdout, corrected team labels; not a fresh independent benchmark.',
                localization_note='No localization accuracy claimed for corrected positives without verified timestamps.')
    joblib.dump(model,model_path)
    meta=json.loads((root/'models/fullframe_classifier.json').read_text())
    meta.update(pipeline_sha256=sha256(model_path),threshold=threshold,classifier='weakly_supervised_event_categories',
                train_ids=[r['row']['id'] for r in training],train_sha256=[r['row']['sha256'] for r in training],validation_ids=[],
                event_types=[str(c) for c in model.estimator.classes_ if c!='normal'],
                annotation_source='Team descriptions, confirmed by user; original Excel preserved',
                annotations_sha256=sha256(root/'data/raw/team_annotations.json'),weak_positive_ids=[r['row']['id'] for r in training if r['weak']],
                limitations='Experimental categories, few videos per type, weak localization; no legal source bindings; CPU.',
                test_used_for_training=False,threshold_objective='fixed_0.55_not_calibrated')
    for field in ['validation_metrics','candidates','alpha']: meta.pop(field,None)
    write_json(model_path.with_suffix('.json'),meta)
    write_json(root/'artifacts/typed_evaluation.json',report)
    print(json.dumps(dict(training_videos=len(training),event_types=meta['event_types'],report=report),ensure_ascii=True,indent=2))

if __name__=='__main__':
    from ml.typed import main as run
    run()