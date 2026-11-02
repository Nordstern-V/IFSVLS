#!/usr/bin/env python3
"""Read-only PYNQ diagnosis; Python 3.8+, standard library only.

Copy this file to the target before running it there. Example:
  python3 collect_pynq_state.py --project-root /path/to/project \
      --bitstream artifacts/design.bit --output /tmp/pynq-state.json

No PYNQ import, Overlay download, MMIO, IIC scan, GPIO reset, sudo,
network connection, package installation or boot-file modification occurs.
Only the explicitly requested JSON output is written. Existing output is
refused unless --overwrite is provided. --dmesg adds a filtered log probe.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


MAX_TEXT = 16384
MAX_XML = 16 * 1024 * 1024


def read_text(path, limit=MAX_TEXT):
    try:
        with Path(path).open("rb") as handle:
            raw = handle.read(limit + 1)
        return {"status": "ok", "text": raw[:limit].decode("utf-8", "replace"),
                "truncated": len(raw) > limit}
    except OSError as error:
        return {"status": "unavailable", "error": str(error)}


def file_info(path):
    path = Path(path)
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        stat = path.stat()
        return {"path": str(path), "status": "ok", "bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": digest.hexdigest()}
    except OSError as error:
        return {"path": str(path), "status": "unavailable", "error": str(error)}


def inspect_hwh(path):
    path = Path(path)
    try:
        if path.stat().st_size > MAX_XML:
            return {"status": "skipped", "reason": "HWH exceeds 16 MiB inspection limit"}
        root = ET.parse(str(path)).getroot()
        modules, ranges = [], []
        wanted = {"C_IIC_FREQ", "IIC_FREQ_KHZ", "C_SG_LENGTH_WIDTH",
                  "C_M_AXI_MM2S_DATA_WIDTH", "C_M_AXI_S2MM_DATA_WIDTH",
                  "PCW_USB0_PERIPHERAL_ENABLE", "PCW_USB0_USB0_IO", "PCW_USB0_RESET_IO"}
        for element in root.iter():
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "MODULE":
                attrs = element.attrib
                params = {}
                for child in element.iter():
                    if child.tag.rsplit("}", 1)[-1] == "PARAMETER":
                        name = child.attrib.get("NAME")
                        if name in wanted:
                            params[name] = child.attrib.get("VALUE")
                modules.append({"instance": attrs.get("INSTANCE") or attrs.get("FULLNAME"),
                                "type": attrs.get("MODTYPE"), "vlnv": attrs.get("VLNV"),
                                "parameters": params})
            elif tag == "MEMRANGE":
                ranges.append({k: element.attrib.get(k) for k in
                               ("INSTANCE", "BASEVALUE", "HIGHVALUE", "MASTERBUSINTERFACE")})
        return {"status": "ok", "module_count": len(modules), "modules": modules[:128],
                "address_ranges": ranges[:128],
                "truncated": len(modules) > 128 or len(ranges) > 128}
    except (OSError, ET.ParseError, ValueError) as error:
        return {"status": "unavailable", "error": str(error)}


def run_command(argv, timeout=5.0, limit=MAX_TEXT):
    if not shutil.which(argv[0]):
        return {"command": argv, "status": "not_found"}
    try:
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=timeout, check=False, shell=False)
        return {"command": argv, "status": "ok" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout[:limit].decode("utf-8", "replace"),
                "stderr": result.stderr[:limit].decode("utf-8", "replace"),
                "truncated": len(result.stdout) > limit or len(result.stderr) > limit}
    except subprocess.TimeoutExpired:
        return {"command": argv, "status": "timeout", "timeout_seconds": timeout}
    except OSError as error:
        return {"command": argv, "status": "unavailable", "error": str(error)}


def project_state(root, bitstream=None):
    root = Path(root).resolve()
    result = {"root": str(root), "versions": {}, "hardware_files": [], "models": []}
    for relative in ("pynq_app/pefsl_touch_ui.py", "touch_deploy/touch_driver.py"):
        probe = read_text(root / relative, 65536)
        if probe["status"] == "ok":
            matches = re.findall(r"(?m)^\s*(APP_VERSION|DRIVER_VERSION)\s*=\s*['\"]([^'\"]+)['\"]", probe["text"])
            if matches:
                result["versions"][relative] = dict(matches)
    if bitstream:
        bit_path = Path(bitstream)
        if not bit_path.is_absolute():
            bit_path = root / bit_path
        candidates = [bit_path.resolve()]
    else:
        candidates = sorted(set(list(root.glob("*.bit")) + list((root / "artifacts").glob("*.bit"))))
    result["bitstream_count"] = len(candidates)
    for bit_path in candidates[:16]:
        hwh_path = bit_path.with_suffix(".hwh")
        result["hardware_files"].append({"bitstream": file_info(bit_path),
                                         "same_stem_hwh": file_info(hwh_path),
                                         "hwh_metadata": inspect_hwh(hwh_path)})
    result["pairing_note"] = "Same stem and hashes identify local files; they do not prove build pairing or current PL state."
    model_files = sorted(set(list(root.glob("*.tmodel")) + list((root / "tcu_deploy").glob("*.tmodel"))))
    for model_path in model_files[:8]:
        item = file_info(model_path)
        probe = read_text(model_path, 2 * 1024 * 1024)
        if probe["status"] == "ok" and not probe["truncated"]:
            try:
                model = json.loads(probe["text"])
                if isinstance(model, dict):
                    item["declared_inputs"] = model.get("inputs")
                    item["declared_outputs"] = model.get("outputs")
            except (ValueError, TypeError) as error:
                item["metadata_error"] = str(error)
        result["models"].append(item)
    return result


def input_summary(text):
    """Keep names, handlers and capability bitmaps; omit Phys/Uniq identifiers."""
    return "\n".join(line for line in text.splitlines()
                     if not line or line.startswith(("I:", "N:", "H:", "B:")))


def linux_state(timeout, include_dmesg=False):
    if not sys.platform.startswith("linux"):
        return {"status": "not_applicable", "reason": "Run on the Linux board to collect board USB/input/DT state."}
    result = {"status": "collected", "commands": [run_command(["uname", "-a"], timeout),
                                                  run_command(["lsusb"], timeout),
                                                  run_command(["lsusb", "-t"], timeout)]}
    inputs = read_text("/proc/bus/input/devices", 65536)
    if inputs["status"] == "ok":
        inputs["text"] = input_summary(inputs["text"])
    result["input_devices"] = inputs
    try:
        result["input_events"] = sorted(str(p) for p in Path("/dev/input").glob("event*"))
        result["keyboard_links"] = sorted(str(p) for p in Path("/dev/input/by-id").glob("*-event-kbd"))
    except OSError as error:
        result["input_error"] = str(error)
    usb_root = Path("/sys/bus/usb/devices")
    try:
        result["usb_sysfs"] = {"status": "ok", "devices": sorted(p.name for p in usb_root.iterdir())[:128]}
    except OSError as error:
        result["usb_sysfs"] = {"status": "unavailable", "error": str(error)}
    bindings = []
    try:
        for device in sorted(Path("/sys/bus/platform/devices").glob("*usb*"))[:32]:
            driver = device / "driver"
            bindings.append({"device": device.name,
                             "driver": str(driver.resolve()) if driver.is_symlink() else None})
    except OSError as error:
        result["platform_binding_error"] = str(error)
    result["usb_platform_bindings"] = bindings
    dt_root = Path("/proc/device-tree")
    result["device_tree"] = {"root_exists": dt_root.exists(), "usb_nodes": []}
    if dt_root.exists():
        try:
            for node in sorted(dt_root.rglob("usb@*"))[:16]:
                props = {}
                for name in ("status", "compatible", "dr_mode", "phy_type", "usb-phy", "phys", "phy-names"):
                    prop = node / name
                    if name in ("usb-phy", "phys"):
                        try:
                            props[name] = {"status": "ok", "hex": prop.read_bytes()[:128].hex()}
                        except OSError as error:
                            props[name] = {"status": "unavailable", "error": str(error)}
                    else:
                        props[name] = read_text(prop, 1024)
                        if props[name]["status"] == "ok":
                            props[name]["text"] = props[name]["text"].replace("\x00", " | ").strip(" | ")
                result["device_tree"]["usb_nodes"].append({"path": str(node), "properties": props})
        except OSError as error:
            result["device_tree"]["error"] = str(error)
    if include_dmesg:
        log = run_command(["dmesg"], timeout, 2 * 1024 * 1024)
        if "stdout" in log:
            selected = [line for line in log["stdout"].splitlines()
                        if re.search(r"usb|ci_hdrc|ehci|ulpi|hid|vdma|iic|axi.*dma|cma", line, re.I)]
            log["stdout"] = "\n".join(selected)[-65536:]
        result["filtered_dmesg"] = log
    return result


def collect(root, bitstream=None, timeout=5.0, include_dmesg=False):
    try:
        pynq_version = importlib.metadata.version("pynq")
    except importlib.metadata.PackageNotFoundError:
        pynq_version = None
    return {"schema": "pynq-engineering-state-v1",
            "collected_utc": datetime.now(timezone.utc).isoformat(),
            "diagnostic_only": True, "hardware_io_performed": False,
            "runtime": {"platform": sys.platform, "machine": platform.machine(),
                        "python": sys.version, "executable": sys.executable,
                        "pointer_bits": 64 if sys.maxsize > 2**32 else 32,
                        "pynq_distribution_version": pynq_version},
            "project": project_state(root, bitstream),
            "linux": linux_state(timeout, include_dmesg)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", default=".", help="Actual project root on this machine")
    parser.add_argument("--bitstream", help="Local bitstream; relative paths use project root")
    parser.add_argument("--output", required=True, help="JSON report path; parent directory must exist")
    parser.add_argument("--timeout", type=float, default=5.0, help="Per-command timeout in seconds")
    parser.add_argument("--dmesg", action="store_true", help="Include filtered kernel log if readable")
    parser.add_argument("--overwrite", action="store_true", help="Replace the explicit output report")
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()
    output = Path(args.output).resolve()
    if not root.is_dir():
        parser.error("project root is not a directory")
    if not output.parent.is_dir():
        parser.error("output parent directory does not exist")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if output.exists() and not args.overwrite:
        parser.error("output already exists; choose a fresh path or explicitly use --overwrite")
    protected = {p.resolve() for pattern in ("*.bit", "*.hwh", "*.tmodel")
                 for base in (root, root / "artifacts", root / "tcu_deploy")
                 for p in base.glob(pattern)}
    if args.bitstream:
        bit_path = Path(args.bitstream)
        bit_path = bit_path if bit_path.is_absolute() else root / bit_path
        protected.update((bit_path.resolve(), bit_path.with_suffix(".hwh").resolve()))
    if output in protected or output.suffix.lower() != ".json":
        parser.error("output must be a separate .json report, not a hardware or model file")
    report = collect(root, args.bitstream, args.timeout, args.dmesg)
    try:
        with output.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except OSError as error:
        parser.error(str(error))
    print("Report: {}".format(output))
    print("Runtime: {} / Python {} / {}-bit".format(sys.platform, platform.python_version(), report["runtime"]["pointer_bits"]))
    print("Bitstreams inspected: {}; Linux state: {}".format(report["project"]["bitstream_count"], report["linux"]["status"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
