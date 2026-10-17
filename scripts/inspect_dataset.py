import argparse
from pathlib import Path

import pandas as pd

from ml.dataset import ForteError, key, timestamp, write_json, sha256
from ml.preprocessing import probe


def inspect(labels, videos):
    if not labels.is_file():
        raise ForteError(f'Нет Excel: {labels}')
    if not videos.is_dir():
        raise ForteError(f'Нет каталога: {videos}')
    table = pd.read_excel(labels, engine='openpyxl').where(lambda df: df.notna(), None)
    table.columns = [str(c).strip().casefold() for c in table.columns]
    expected = ['наименование файла', 'таймстемп', 'нарушение / без нарушения']
    if not set(expected).issubset(table.columns):
        raise ForteError(f'Ожидаются колонки: {expected}')
    files = {}
    for path in videos.iterdir():
        if path.is_file() and path.suffix.casefold() == '.mov':
            files.setdefault(key(path.name), []).append(path)
    result = dict(rows=0, positive=0, negative=0, missing_videos=[], invalid_timestamps=[], errors=[], videos=[])
    seen = set()
    for index, row in table.iterrows():
        if row.isna().all():
            continue
        result['rows'] += 1
        name, raw_time, label = [row[c] for c in expected]
        item = dict(row=int(index)+2, video_id=key(name), original_name=str(name))
        lab = str(label).strip().casefold()
        mapping = {'нарушение': 1, 'без нарушения': 0, '1': 1, '0': 0}
        if lab not in mapping:
            result['errors'].append(dict(**item, error='Неизвестная метка'))
        else:
            item['label'] = mapping[lab]
            result['positive' if mapping[lab] else 'negative'] += 1
        if item['video_id'] in seen:
            result['errors'].append(dict(**item, error='Повтор video ID'))
        seen.add(item['video_id'])
        try:
            item['timestamp_sec'] = timestamp(None if pd.isna(raw_time) else raw_time)
            if item.get('label') == 1 and item['timestamp_sec'] is None:
                raise ForteError('Нет timestamp положительного видео')
            if item.get('label') == 0 and item['timestamp_sec'] is not None:
                raise ForteError('Timestamp у отрицательного видео')
        except ForteError as exc:
            result['invalid_timestamps'].append(dict(**item, error=str(exc)))
        matches = files.get(item['video_id'], [])
        if not matches:
            result['missing_videos'].append(item)
        elif len(matches) > 1:
            result['errors'].append(dict(**item, error='Неоднозначное соответствие X/Х'))
        else:
            try:
                item.update(file=matches[0].name, **probe(matches[0]))
                ts = item.get('timestamp_sec')
                if ts is not None and ts >= item['duration_seconds']:
                    result['invalid_timestamps'].append(dict(**item, error='Timestamp вне видео'))
            except ForteError as exc:
                result['errors'].append(dict(**item, error=str(exc)))
        result['videos'].append(item)
    result['unlabeled_videos'] = [p.name for k, group in files.items() if k not in seen for p in group]
    hashes = {}
    for group in files.values():
        for path in group:
            hashes.setdefault(sha256(path), []).append(path.name)
    result['identical_content_groups'] = [names for names in hashes.values() if len(names) > 1]
    result['scope'] = 'Metadata inspection only; no semantic claims, no full corruption scan.'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=Path, default=Path('data/raw/open_labels.xlsx'))
    parser.add_argument('--videos', type=Path, default=Path('data/videos'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/dataset_inspection.json'))
    args = parser.parse_args()
    try:
        result = inspect(args.labels, args.videos)
        write_json(args.output, result)
        import json
        print(json.dumps(result, ensure_ascii=True, indent=2))
        if any(result[k] for k in ('missing_videos', 'invalid_timestamps', 'errors')):
            parser.exit(2, 'Dataset validation errors; see report.\n')
    except (ForteError, OSError, ValueError) as exc:
        parser.exit(2, ascii(str(exc))+'\n')


if __name__ == '__main__':
    main()
