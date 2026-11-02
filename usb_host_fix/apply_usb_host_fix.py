#!/usr/bin/env python3
"""Check/install/restore the exact-image USB Host DT fix (Python 3.8+)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


HERE = Path(__file__).resolve().parent
BACKUP_SUFFIX = ".before-usb-host-20261101"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def install_atomically(source, target, expected):
    """Copy, fsync and verify before replacing the named boot image."""
    fd, temporary_name = tempfile.mkstemp(
        prefix=target.name + ".usb-host-", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as incoming:
            shutil.copyfileobj(incoming, output)
            output.flush()
            os.fsync(output.fileno())
        if digest(temporary) != expected:
            raise RuntimeError("temporary file checksum mismatch; original retained")
        os.chmod(str(temporary), target.stat().st_mode & 0o777)
        os.replace(str(temporary), str(target))
        os.sync()
        if digest(target) != expected:
            raise RuntimeError("installed file checksum mismatch; restore from backup")
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boot-image", default="/boot/image.ub")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--apply", action="store_true", help="back up and install")
    actions.add_argument("--restore", action="store_true", help="restore verified backup")
    actions.add_argument("--check", action="store_true", help="read-only check (default)")
    args = parser.parse_args(argv)
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    original_hash = manifest["original_image_sha256"]
    patched_hash = manifest["patched_image_sha256"]
    patch = HERE / manifest["patched_image_file"]
    requested = Path(args.boot_image)
    if requested.name.lower() != "image.ub":
        raise RuntimeError("--boot-image must identify the existing image.ub boot file")
    if not requested.is_file():
        raise RuntimeError(
            "boot image not found: {}. Check lsblk -f and the boot mount first".format(requested)
        )
    boot = requested.resolve()
    backup = boot.with_name(boot.name + BACKUP_SUFFIX)
    current_hash = digest(boot)
    print("Boot image: {}".format(boot), flush=True)
    print("Current SHA256: {}".format(current_hash), flush=True)
    if current_hash not in (original_hash, patched_hash):
        raise RuntimeError(
            "this boot image differs from the inspected original and fixed image; "
            "no files changed. Provide this image.ub for a matching patch"
        )
    if args.restore:
        if current_hash == original_hash:
            print("Original image is already installed; no files changed")
            return 0
        if not backup.is_file() or digest(backup) != original_hash:
            raise RuntimeError("verified original backup not found: {}".format(backup))
        if os.geteuid() != 0:
            raise RuntimeError("restoring requires sudo")
        install_atomically(backup, boot, original_hash)
        print("Original boot image restored. Reboot when ready.")
        return 0
    if not patch.is_file() or digest(patch) != patched_hash:
        raise RuntimeError("fixed image file is missing or its checksum differs from manifest")
    if not args.apply:
        print("Fixed image checksum: PASS")
        print("Boot state: {}".format("original" if current_hash == original_hash else "USB Host fix"))
        print("Check complete; no files changed.")
        return 0
    if current_hash == patched_hash:
        print("USB Host fix is already installed; no files changed")
        return 0
    if os.geteuid() != 0:
        raise RuntimeError("installing requires sudo")
    if backup.exists() and (not backup.is_file() or digest(backup) != original_hash):
        raise RuntimeError("backup path already exists with different contents: {}".format(backup))
    required = patch.stat().st_size + (0 if backup.exists() else boot.stat().st_size)
    if shutil.disk_usage(str(boot.parent)).free < required + 1024 * 1024:
        raise RuntimeError("insufficient space for backup and verified temporary image")
    if not backup.exists():
        with boot.open("rb") as incoming, backup.open("xb") as output:
            shutil.copyfileobj(incoming, output)
            output.flush()
            os.fsync(output.fileno())
        os.sync()
    if digest(backup) != original_hash:
        raise RuntimeError("backup checksum mismatch; original boot image retained")
    install_atomically(patch, boot, patched_hash)
    print("USB Host DT fix installed successfully.", flush=True)
    print("Verified backup: {}".format(backup), flush=True)
    print("Run sudo reboot to activate. No reboot was performed by this script.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print("ERROR: {}".format(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
