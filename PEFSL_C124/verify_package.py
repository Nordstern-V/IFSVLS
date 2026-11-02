"""Verify shipped files using the version manifest; no camera or learning writes."""
from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parent
version=json.loads((root/'VERSION.json').read_text(encoding='utf-8'))
failed=[]
for name,expected in version['files_sha256'].items():
    path=root/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        failed.append(name)
if failed:
    raise SystemExit('Missing or changed files: '+', '.join(failed))
print('OK: C124 PC + snapshot filename fix; '+str(len(version['files_sha256']))+' files verified.')
