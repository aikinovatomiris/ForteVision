"""Create local configuration once, without printing credentials."""
from pathlib import Path
import secrets

def initialize(root):
    root=Path(root)
    content=(root/'.env.example').read_text(encoding='utf-8')
    placeholder='replace-with-a-long-random-local-secret'
    if content.count(placeholder)!=1:
        raise ValueError('Expected one API-key placeholder')
    content=content.replace(placeholder,secrets.token_urlsafe(32))
    with (root/'.env').open('x',encoding='utf-8') as target:
        target.write(content)

def main():
    root=Path(__file__).resolve().parents[1]
    try:initialize(root)
    except FileExistsError:
        print('Existing .env preserved; no settings changed.')
    else:print('Created .env with a random local API key. Credentials were not printed.')

if __name__=='__main__':main()
