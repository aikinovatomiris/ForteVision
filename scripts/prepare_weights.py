"""The ONLY model-download command. Run explicitly before going offline."""
import argparse
import urllib.request
from pathlib import Path

from ml.dataset import ForteError, sha256, write_json

URL = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'


def main():
    parser = argparse.ArgumentParser(description='Download official ResNet18 weights; review LICENSES.md for weight/data terms.')
    parser.add_argument('--output', type=Path, default=Path('models/resnet18-f37072fd.pth'))
    args = parser.parse_args()
    temporary = args.output.with_suffix('.download')
    try:
        if args.output.exists():
            if not sha256(args.output).startswith('f37072fd'):
                raise ForteError('Уже существующие веса имеют неверный SHA-256')
            print('Local weights already present and verified.')
            return
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open('wb') as target:
            while block := response.read(1024*1024):
                target.write(block)
        digest = sha256(temporary)
        if not digest.startswith('f37072fd'):
            raise ForteError('SHA-256 загруженных весов не соответствует официальному имени')
        temporary.replace(args.output)
        write_json(args.output.with_suffix('.source.json'), dict(url=URL, sha256=digest, architecture='resnet18', weights='IMAGENET1K_V1', license_review='See LICENSES.md; commercial suitability of pretrained weights is not certified.'))
        print(f'Weights saved: {args.output}')
    except (OSError, ForteError) as exc:
        parser.exit(2, ascii(str(exc))+'\n')
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
