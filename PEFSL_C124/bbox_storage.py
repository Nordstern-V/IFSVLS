"""Publish complete archive snapshots without replacing an existing file."""
import os
import re
import tempfile
import uuid
from datetime import datetime

import numpy as np


def _snapshot_prefix(stem, directory, extension):
    # A previous snapshot becomes the next save target. Remove only our own
    # trailing suffixes instead of accumulating the full save history.
    prefix = re.sub(r'(?:_\d{8}_\d{6}_[0-9a-f]{12})+$', '',
                    os.path.basename(stem), flags=re.IGNORECASE) or 'learning'
    limit = 48
    if os.name == 'nt':
        # Leave room below the ordinary Windows path limit, including the
        # optional background sidecar suffix. UUID/time suffix is 29 chars.
        limit = min(limit, 240 - len(directory) - 1 - 29 - len(extension))
        if limit < 1:
            raise OSError('Save directory path is too long; choose a shorter directory')
    return prefix[:limit]


def save_archive_snapshot(path, payload):
    """Return (success, error, actual_path); use an exact destination suffix.

    A same-directory temporary file is fully written before an atomic hard
    link publishes it. Linking refuses an existing destination on both Windows
    and Linux. Filesystems without hard-link support return a save error.
    """
    temporary = None
    try:
        target = os.path.abspath(os.fspath(path))
        directory = os.path.dirname(target)
        stem, extension = os.path.splitext(target)
        prefix = None
        use_requested = (len(os.path.basename(target).encode('utf-8')) <= 240
                         and (os.name != 'nt' or len(target) <= 240))
        with tempfile.NamedTemporaryFile(mode='wb', prefix='.pefsl-', suffix='.tmp',
                                         dir=directory, delete=False) as stream:
            temporary = stream.name
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(16):
            if attempt == 0 and use_requested:
                destination = target
            else:
                if prefix is None:
                    prefix = _snapshot_prefix(stem, directory, extension)
                destination = os.path.join(directory, '{}_{}_{}{}'.format(
                    prefix, datetime.now().strftime('%Y%m%d_%H%M%S'),
                    uuid.uuid4().hex[:12], extension))
            try:
                os.link(temporary, destination)
            except FileExistsError:
                continue
            return True, '', destination
        return False, 'Cannot allocate a new snapshot filename', None
    except (OSError, ValueError, TypeError) as error:
        return False, str(error), None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)  # Only the private .tmp file, never an archive.
            except OSError:
                pass
