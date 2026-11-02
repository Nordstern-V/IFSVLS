"""Start full C124 learning/recognition with package-local data paths."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
data=root/'data'
diagnostics=root/'diagnostics'
data.mkdir(exist_ok=True)
diagnostics.mkdir(exist_ok=True)
archives=list(data.glob('learning*.npz'))
archive=max(archives,key=lambda p:(p.stat().st_mtime_ns,p.name)) if archives else data/'learning.npz'
print('PEFSL C124 PC / snapshot filename fix 2026-10-25',flush=True)
print('Learning archive:',archive,flush=True)
print('Press L in the app to load an existing archive, then B to recognize.',flush=True)
command=[sys.executable,str(root/'bbox_demo.py'),'--save-path',str(archive),
         '--diag-dir',str(diagnostics),*sys.argv[1:]]
raise SystemExit(subprocess.call(command,cwd=root))
