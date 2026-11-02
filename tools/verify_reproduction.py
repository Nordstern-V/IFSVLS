#!/usr/bin/env python3
"""Verify the selected project files without requiring PYNQ or Vivado."""
import hashlib
import json
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'reproduction_manifest.json').read_text(encoding='utf-8'))
    errors = []
    entries = manifest['files']
    for item in entries:
        path = root / item['path']
        if not path.is_file():
            errors.append('Missing: ' + item['path'])
            continue
        data = path.read_bytes()
        if len(data) != item['size'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            errors.append('Content mismatch: ' + item['path'])
        if path.suffix == '.py':
            try:
                compile(data, str(path), 'exec')
            except (SyntaxError, ValueError) as error:
                errors.append('Python syntax: {}: {}'.format(item['path'], error))

    for model in (root / 'tcu_deploy').glob('*.tmodel'):
        metadata = json.loads(model.read_text(encoding='utf-8'))
        for component in [metadata['prog']] + metadata.get('consts', []):
            target = model.parent / component['file_name']
            if not target.is_file():
                errors.append('Missing model component: ' + str(target.relative_to(root)))
    for ext in ('.bit', '.hwh'):
        if not (root / 'artifacts' / ('pefsl_mix_plus_touch' + ext)).is_file():
            errors.append('Missing paired overlay: ' + ext)

    if errors:
        print('\n'.join(errors))
        return 1
    print('PASS: {} selected files; SHA256, Python syntax, overlay pair and model references verified.'.format(len(entries)))
    print('Board execution and Vivado synthesis must be verified in their target environments.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
