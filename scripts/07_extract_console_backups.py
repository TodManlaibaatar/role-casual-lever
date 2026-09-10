"""Recover checksummed artifacts from the local Colab console log."""
import argparse
import base64
import gzip
import hashlib
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('log', type=Path)
parser.add_argument('--output', type=Path, default=Path('results/recovered'))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
count = 0
for line in args.log.read_text(errors='replace').splitlines():
    if not line.startswith('MATS_BACKUP '):
        continue
    try:
        _, name, digest, encoded = line.split(' ', 3)
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Invalid artifact name')
        data = gzip.decompress(base64.b64decode(encoded, validate=True))
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('Checksum mismatch')
        (args.output / name).write_bytes(data)
        count += 1
    except Exception as error:
        print('Skipped incomplete or invalid backup:', str(error))
print('Recovered', count, 'checkpoints into', args.output.resolve())
if count == 0:
    raise SystemExit('No valid backups found; inspect the console log.')
