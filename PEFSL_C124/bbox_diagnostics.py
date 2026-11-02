"""Diagnostic summaries only: no acceptance decisions or learning-file writes."""
import hashlib
import os
from pathlib import Path
import numpy as np


def _file_hash(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


SOURCE_HASHES = {p.name: _file_hash(p) for p in Path(__file__).parent.glob('bbox*.py')}


def _summary(value):
    if isinstance(value, np.generic):return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):return value
    if isinstance(value, (list, tuple)) and len(value)<=8:
        return [_summary(v) for v in value]
    return None


def identity_attempt(task, pattern, reference, status, elapsed_ms, remaining_ms):
    """Copy scalar CURRENT results; exclude images, learned hints and point arrays."""
    result=dict(task=task,status=status,elapsed_ms=round(elapsed_ms,3),
                remaining_ms=round(remaining_ms,3),reference=round(float(reference),4))
    if pattern is None:return result
    result['pattern']={k:_summary(pattern[k]) for k in ('similarity','patch','reference_extent_iou') if k in pattern}
    fields=('verified','reference','reference_view','rotation','flip','detector','inliers','matches',
            'ratio','source_coverage','coverage','extent','extent_iou','mean_ncc','median_error',
            'correlation','aligned_hog','supported_tiles','valid_fraction')
    for kind in ('geometry','dense','spatial'):
        if kind in pattern:
            result['pattern'][kind]={k:_summary(pattern[kind][k]) for k in fields if k in pattern[kind]}
    return result


def learning_fingerprint(data):
    """Include live learning arrays, excluding derived caches and camera pixels."""
    digest=hashlib.sha256()
    def add(value):
        if isinstance(value,np.ndarray):
            digest.update(str((value.dtype.str,value.shape)).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value,(list,tuple)):
            digest.update(str(len(value)).encode()+b'[')
            for v in value:add(v)
            digest.update(b']')
        else:digest.update(repr(value).encode('utf-8')+b'\0')
    for name in ('labels','shots','class_fracs','class_ars','thumbs','visual_thumbs','reference_views',
                 'generic_classes','class_hists','class_bgs','class_shapes','neg_feats','_legacy_mean'):
        digest.update(name.encode());add(getattr(data,name,None))
    return digest.hexdigest()


def diagnostic_context(app):
    archive=Path(app.args.save_path).resolve()
    current={name:_file_hash(Path(__file__).parent/name) for name in SOURCE_HASHES}
    bg=getattr(app,'bg_model',None)
    return dict(pid=os.getpid(),learning_path=str(archive),
                archive_file_sha256=_file_hash(archive),
                live_learning_sha256=learning_fingerprint(app.data),
                labels=list(app.data.labels),shots=[app.data.total_shots(lab) for lab in app.data.labels],
                search_epoch=app._search_epoch,
                capture_time=getattr(app,'_capture_time',None),
                current_frame_available=getattr(app,'raw_frame',None) is not None,
                search_error=getattr(app,'_search_last_error',None),
                search_error_streak=getattr(app,'_search_error_streak',0),
                background_sha256=None if bg is None else hashlib.sha256(np.ascontiguousarray(bg).tobytes()).hexdigest(),
                source_sha256_at_import=SOURCE_HASHES.copy(),
                sources_changed_on_disk=[name for name in SOURCE_HASHES if current[name]!=SOURCE_HASHES[name]],
                note='Archive file hash identifies disk contents; live learning fingerprint also changes after unsaved learning.')
