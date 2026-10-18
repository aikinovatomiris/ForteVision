"""Deterministic workflow checks over explicit local visual observations.
No rule treats a missing observation as proof that an action did not occur.
"""

ACTIONS = ('customer_arrives','customer_leaves','document_presented','cash_received','cash_counted',
           'cash_into_storage','storage_closed','storage_open','cashier_leaves','cash_on_counter',
           'cash_added_to_existing_stack','storage_locked','storage_unlocked','document_candidate','key_in_storage_lock')

RULES = [
    ('customer_arrival','Приход клиента','customer_arrives'),
    ('identity_document','Передача документа','document_presented'),
    ('identity_check','Сверка удостоверения и фиксация данных',None),
    ('cash_receipt','Получение наличных','cash_received'),
    ('cash_recount','Пересчёт наличных','cash_counted'),
    ('cash_storage','Уборка наличных в хранилище','cash_into_storage'),
    ('storage_closure','Закрытие хранилища','storage_closed'),
    ('storage_lock','Запирание хранилища на ключ','storage_locked'),
    ('change_return','Возврат сдачи, если она требуется',None),
    ('personal_items','Уборка личных вещей перед уходом',None),
    ('client_separation','Разделение денег разных клиентов',None),
]

def check_workflow(observations):
    obs=sorted(observations,key=lambda o:o['timestamp_sec'])
    findings=[]
    for o in obs:
        action=o['action'];t=o['timestamp_sec']
        if action=='cash_added_to_existing_stack':
            preceding=[p for p in obs if p['timestamp_sec']<=t and p['action'] in ('customer_arrives','customer_leaves')]
            departed=[p for p in preceding if p['action']=='customer_leaves']
            arrived=[p for p in preceding if p['action']=='customer_arrives' and any(d['timestamp_sec']<p['timestamp_sec'] for d in departed)]
            if arrived:
                findings.append(dict(event_type='prior_deposit_outside_storage',type='Возможное смешивание денег разных клиентов',
                    description='Локальная модель отметила смену клиента и добавление наличных к уже лежащей стопке. Проверьте принадлежность стопок разным операциям.',
                    evidence=[departed[-1],arrived[-1],o]))
        if action=='cashier_leaves':
            keys=[p for p in obs if t<=p['timestamp_sec']<=t+15 and p['action']=='key_in_storage_lock']
            if keys:
                findings.append(dict(event_type='key_left_in_lock',type='Ключ может быть оставлен в замке сейфа после ухода',description='Модель отметила уход кассира и ключ в замке хранилища после ухода. Проверьте ключ на исходной записи; это не устанавливает состояние запирания.',evidence=[o,keys[0]]))
            after=[p for p in obs if t<=p['timestamp_sec']<=t+15 and p['action']=='cash_on_counter']
            stored=[p for p in obs if t<=p['timestamp_sec']<=t+15 and p['action']=='cash_into_storage']
            if after and not stored:
                findings.append(dict(event_type='cash_left_unattended',type='Наличные могут быть оставлены без присмотра',
                    description='Локальная модель отметила уход кассира и наличные на рабочей поверхности после ухода. Проверьте последовательность по кадрам.',evidence=[o,after[0]]))
            unlocked=[p for p in obs if t<=p['timestamp_sec']<=t+15 and p['action']=='storage_unlocked']
            locked=[p for p in obs if t<=p['timestamp_sec']<=t+15 and p['action']=='storage_locked']
            if unlocked and not locked:
                findings.append(dict(event_type='storage_left_unlocked',type='Хранилище может быть оставлено незапертым',description='После ухода кассира модель отметила явно незапертый замок. Проверьте механизм замка по исходной записи; закрытая дверца и ключ в замке сами по себе не определяют состояние замка.',evidence=[o,unlocked[0]]))
            opened=[p for p in obs if t<=p['timestamp_sec']<=t+15 and p['action']=='storage_open']
            closed=[p for p in obs if t<=p['timestamp_sec']<=t+15 and p['action']=='storage_closed']
            if opened and not closed:
                findings.append(dict(event_type='storage_left_open',type='Хранилище может быть оставлено открытым',
                    description='Локальная модель отметила уход кассира и открытое хранилище после него. Проверьте, что видна именно дверца хранилища.',evidence=[o,opened[0]]))
        if action=='customer_arrives':
            # Need explicit cash visible before and after the switch, not merely a missing storage action.
            left=[p for p in obs if t-30<=p['timestamp_sec']<t and p['action']=='customer_leaves']
            before=[p for p in obs if t-15<=p['timestamp_sec']<t and p['action']=='cash_on_counter']
            after=[p for p in obs if t<=p['timestamp_sec']<=t+10 and p['action']=='cash_on_counter']
            stored=[p for p in obs if t-15<=p['timestamp_sec']<=t+10 and p['action']=='cash_into_storage']
            if left and before and after and not stored:
                findings.append(dict(event_type='prior_deposit_outside_storage',type='Предыдущие деньги могут оставаться при смене клиента',
                    description='Модель отметила уход клиента, появление следующего и наличные до и после смены. Нужно проверить, что это одна и та же партия денег.',evidence=[before[-1],o,after[0]]))
    unique=[]
    for f in findings:
        peak=f['evidence'][-1]['timestamp_sec']
        if any(e['event_type']==f['event_type'] and abs(e['peak']-peak)<8 for e in unique):continue
        f['peak']=peak;unique.append(f)
    checks=[]
    for ident,title,action in RULES:
        matching=[o for o in obs if action is not None and o['action']==action]
        checks.append(dict(rule_id=ident,title=title,state='observed' if matching else 'unknown',
            timestamps=[o['timestamp_sec'] for o in matching],
            note='Предположение локальной визуальной модели; не юридическое подтверждение.' if matching else 'Действие не подтверждено выбранными кадрами. Это не доказывает его отсутствие.'))
    for check in checks:
        if check['rule_id'] in ('identity_check','change_return','personal_items'):
            check['note']='Текущая модель не устанавливает этот шаг: нужны дополнительные наблюдения или данные операции.'
        if check['rule_id']=='client_separation':
            check['note']='Проверяются признаки оставленных или смешанных денег при смене клиента; отсутствие сигнала не подтверждает соблюдение правила.'
    return unique,checks