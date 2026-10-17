"""Offline inventory of installed package metadata and bundled license notices."""
import importlib.metadata as metadata
import json
import re
from pathlib import Path

import cv2

root = Path('artifacts/licenses')
root.mkdir(parents=True, exist_ok=True)
inventory = []
for dist in sorted(metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
    name = re.sub(r'[^a-zA-Z0-9_.-]', '_', dist.metadata['Name'])
    notices = []
    for f in dist.files or []:
        if any(word in Path(str(f)).name.lower() for word in ('license', 'copying', 'notice')) and str(f).lower().endswith(('.txt', '.md', 'license', 'copying', 'notice', 'license-apache', 'license-mit')):
            source = Path(dist.locate_file(f))
            if source.is_file():
                target = root / name / re.sub(r'[^a-zA-Z0-9_.-]', '_', str(f))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                notices.append(str(target))
    inventory.append(dict(name=name, version=dist.version, license_expression=dist.metadata.get('License-Expression'), license=dist.metadata.get('License'), notices=notices))
(root / 'inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
Path('artifacts/opencv-build.txt').write_text(cv2.getBuildInformation(), encoding='utf-8')
print(f'Inventoried {len(inventory)} installed packages in {root}')
