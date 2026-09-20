#!/usr/bin/env python3
"""Select and safely flash the 1k, 3k, or 6k model to both boards."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import serial


ROOT = Path(__file__).resolve().parent
FLASH_DIRS = (ROOT / "firmware",)
FIRMWARE_NAMES = {
    "1k": ("nuttx_r276_lcdinit_1k.bin", "nuttx_r246_1k_speedtest.bin"),
    "3k": ("nuttx_r276_lcdinit_3k.bin", "nuttx_r271.bin", "nuttx_r269_cunorm_psram_3k.bin", "nuttx_usb_token_pcm_v9_ui_refresh.bin", "nuttx_3k_ui_refresh.bin"),
    "6k": ("nuttx_r276_lcdinit_6k.bin", "nuttx_usb_token_pcm_v11_6k_wholefix.bin", "nuttx_6k_wholefix.bin"),
    "pcassist": ("nuttx_r276_lcdinit_3k.bin", "nuttx_r269_cunorm_psram_3k.bin"),
}
FIRMWARE_SHA256 = {
    "1k": "d4034ea120339b6c325bba0574412b3d8fc1cb8cc606626253774dd24e5ba3cc",
    "3k": "be321d5f1685e4064ac2ba4c6fd3e461bd25e3a53287c7e6cd8a5dad4e5db93b",
    "6k": "37b02371f51984e3af8eb16ed841be2b5923c9ad53d297297214cdc2b9bf6352",
    "pcassist": "be321d5f1685e4064ac2ba4c6fd3e461bd25e3a53287c7e6cd8a5dad4e5db93b",
}
PORTS = ("COM4", "COM23")
STATE_FILE = ROOT / "_pc_voice_rx" / "active_rate.json"
STATE_MAX_AGE_SECONDS = 86400


def firmware_path(rate):
    for name in FIRMWARE_NAMES[rate]:
        for folder in FLASH_DIRS:
            candidate = folder / name
            if candidate.exists():
                return candidate
    return FLASH_DIRS[0] / FIRMWARE_NAMES[rate][0]


def stop_ui():
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'pythonw.exe' -and "
        "($_.CommandLine -match '_pc_safe_demo_ui|_pc_token_relay') } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force "
        "-ErrorAction SilentlyContinue }"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=False)
    time.sleep(0.8)


def launch_ui():
    cmd = Path(ROOT / "SafeDemoUI.cmd")
    subprocess.Popen(["cmd", "/c", str(cmd)], cwd=str(ROOT))


def launch_pcassist():
    cmd = Path(ROOT / "PcAssistRealtime.cmd")
    subprocess.Popen(["cmd", "/c", str(cmd)], cwd=str(ROOT))


def query_rate(port):
    connection = serial.Serial()
    connection.port = port
    connection.baudrate = 115200
    connection.timeout = 0.1
    connection.write_timeout = 2.0
    connection.dtr = False
    connection.rts = False
    try:
        connection.open()
        deadline = time.monotonic() + 12.0
        boot = bytearray()
        while time.monotonic() < deadline:
            data = connection.read(4096)
            if data:
                boot.extend(data)
                if b"nsh>" in boot:
                    break
        else:
            return None

        connection.reset_input_buffer()
        connection.write(b"jixun info\r\n")
        connection.flush()
        deadline = time.monotonic() + 4.0
        response = bytearray()
        while time.monotonic() < deadline:
            data = connection.read(4096)
            if not data:
                continue
            response.extend(data)
            match = re.search(rb"\xe7\xa0\x81\xe7\x8e\x87\s*:\s*([136]k)", response)
            if match and b"nsh>" in response:
                return match.group(1).decode("ascii")
        return None
    except (OSError, serial.SerialException):
        return None
    finally:
        connection.close()


def read_cached_rate():
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if time.time() - float(data.get("updated", 0)) > STATE_MAX_AGE_SECONDS:
            return None
        return data.get("rate")
    except (OSError, ValueError, TypeError):
        return None


def write_cached_rate(rate, digest=None):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps({"rate": rate, "sha256": digest, "updated": time.time()}),
        encoding="utf-8",
    )


def flash(rate, port):
    image = firmware_path(rate)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    expected = FIRMWARE_SHA256.get(rate)
    if expected is not None and digest != expected:
        raise RuntimeError(f"firmware hash mismatch for {rate}: {digest}")
    print(f"[FLASH] {rate} -> {port} sha256={digest}", flush=True)
    cmd = [
        sys.executable, "-m", "esptool",
        "--port", port, "--baud", "921600", "--after", "hard-reset",
        "write-flash", "0x10000", str(image),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"flash failed on {port}")
    return digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rate", choices=("1k", "3k", "6k", "pcassist"))
    parser.add_argument("--keep-ui", action="store_true")
    parser.add_argument("--ensure", action="store_true")
    parser.add_argument("--launch-ui", action="store_true")
    parser.add_argument("--launch-pcassist", action="store_true")
    args = parser.parse_args()

    image = firmware_path(args.rate)
    if not image.exists():
        raise SystemExit(f"missing firmware: {image}")

    expected_rate = "3k" if args.rate == "pcassist" else args.rate
    if args.ensure:
        cached_rate = read_cached_rate()
        if cached_rate == expected_rate:
            print(f"[RATE] target={expected_rate} cached=active", flush=True)
            if args.launch_pcassist:
                launch_pcassist()
            elif args.launch_ui:
                launch_ui()
            print(f"[DONE] rate={args.rate} already active (cached)", flush=True)
            return 0
        detected = {port: query_rate(port) for port in PORTS}
        print(f"[RATE] target={expected_rate} detected={detected}", flush=True)
        if all(rate == expected_rate for rate in detected.values()):
            write_cached_rate(expected_rate)
            if args.launch_pcassist:
                launch_pcassist()
            elif args.launch_ui:
                launch_ui()
            print(f"[DONE] rate={args.rate} already active (no flash)", flush=True)
            return 0

    if not args.keep_ui:
        stop_ui()
    digest = flash(args.rate, PORTS[0])
    flash(args.rate, PORTS[1])
    if digest != hashlib.sha256(image.read_bytes()).hexdigest():
        raise RuntimeError("firmware changed during flashing")
    write_cached_rate(expected_rate, digest)
    if args.launch_pcassist:
        launch_pcassist()
    elif args.launch_ui:
        launch_ui()
    print(f"[DONE] rate={args.rate} sha256={digest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
