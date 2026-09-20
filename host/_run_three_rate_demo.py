#!/usr/bin/env python3
"""Show the PC UI and run the 1k -> 3k -> 6k dual-board demo in sequence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FLASH_DIR = ROOT / "firmware"
TRIGGER = ROOT / "_pc_voice_rx" / "rate_sequence_trigger.txt"
DONE = TRIGGER.with_suffix(".done.json")
UI = ROOT / "_pc_safe_demo_ui.py"

RATES = [
    ("1k", FLASH_DIR / "nuttx_r276_lcdinit_1k.bin", "d4034ea120339b6c325bba0574412b3d8fc1cb8cc606626253774dd24e5ba3cc"),
    ("3k", FLASH_DIR / "nuttx_r276_lcdinit_3k.bin", "be321d5f1685e4064ac2ba4c6fd3e461bd25e3a53287c7e6cd8a5dad4e5db93b"),
    ("6k", FLASH_DIR / "nuttx_r276_lcdinit_6k.bin", "37b02371f51984e3af8eb16ed841be2b5923c9ad53d297297214cdc2b9bf6352"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ui_running() -> bool:
    cmd = [
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'pythonw.exe' -and "
        "$_.CommandLine -match '_pc_safe_demo_ui.py' } | Select-Object -First 1 -ExpandProperty ProcessId",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return bool(result.stdout.strip())


def launch_ui() -> None:
    if ui_running():
        print("[UI] already running", flush=True)
        return
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.exists():
        pythonw = Path(sys.executable)
    env = None
    proc = subprocess.Popen(
        [str(pythonw), str(UI), "--trigger-file", str(TRIGGER)],
        cwd=str(ROOT),
        creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
        env=env,
    )
    print(f"[UI] launched pid={proc.pid}", flush=True)
    time.sleep(2.0)


def flash(rate: str, port: str, image: Path) -> None:
    print(f"[FLASH] {rate} {port}", flush=True)
    result = subprocess.run(
        [
            sys.executable, "-m", "esptool", "--port", port, "--baud", "921600",
            "--after", "hard-reset", "write-flash", "0x10000", str(image),
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        tail = "\n".join((result.stdout or "").splitlines()[-20:])
        raise RuntimeError(f"flash failed on {port}\n{tail}")
    if "Hash of data verified." not in (result.stdout or ""):
        raise RuntimeError(f"flash verification missing on {port}")
    print(f"[FLASH] {rate} {port} verified", flush=True)


def trigger_rate(rate: str) -> dict:
    DONE.unlink(missing_ok=True)
    TRIGGER.write_text(f"{rate}\n{time.time():.6f}\n", encoding="utf-8")
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if DONE.exists():
            try:
                data = json.loads(DONE.read_text(encoding="utf-8"))
            except Exception:
                time.sleep(0.2)
                continue
            if data.get("rate") == rate:
                print(
                    f"[DEMO] {rate} rc={data.get('rc')} played={data.get('played')} "
                    f"fragments={data.get('token_fragments')} relayed={data.get('relayed_groups')}",
                    flush=True,
                )
                return data
        time.sleep(0.25)
    raise TimeoutError(f"{rate} relay did not finish within 120s")


def main() -> int:
    for rate, image, expected in RATES:
        if not image.exists():
            raise SystemExit(f"missing {rate} firmware: {image}")
        actual = sha256(image)
        if actual != expected:
            raise SystemExit(f"hash mismatch for {rate}: {actual}")

    # Remove only relay processes; keep any existing showcase window alive.
    subprocess.run([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|pythonw' -and "
        "$_.CommandLine -match '_pc_token_relay.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }",
    ], check=False)
    launch_ui()

    completed = []
    for rate, _image, _ in RATES:
        try:
            result = trigger_rate(rate)
            if result.get("rc") == 0:
                completed.append(rate)
            else:
                print(f"[WARN] {rate} relay failed with rc={result.get('rc')}", flush=True)
        except Exception as exc:
            print(f"[WARN] {rate} stage failed: {exc}", flush=True)
        time.sleep(2.0)

    print(f"[DONE] demo sequence completed: {', '.join(completed)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
