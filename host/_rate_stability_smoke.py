#!/usr/bin/env python3
"""Run repeated dual-board relay cycles for 1k/3k/6k stability checks."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import _switch_codec_rate as rates


ROOT = Path(__file__).resolve().parent
RELAY = ROOT / "_pc_token_relay.py"
WAV = ROOT / "_demo_assets" / "ai_token_demo_cn.wav"
OUT_DIR = ROOT / "_pc_voice_rx" / "rate_stability"


def flash_rate(rate):
    image = rates.firmware_path(rate)
    expected = rates.FIRMWARE_SHA256[rate]
    digest = subprocess.check_output(
        [sys.executable, "-c", "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())", str(image)],
        text=True,
    ).strip()
    if digest != expected:
        raise RuntimeError(f"{rate} image hash mismatch: {digest}")
    for port in rates.PORTS:
        result = subprocess.run(
            [sys.executable, "-m", "esptool", "--port", port, "--baud", "921600",
             "--after", "hard-reset", "write-flash", "0x10000", str(image)],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0 or "Hash of data verified." not in result.stdout:
            raise RuntimeError(f"flash failed on {port} for {rate}")
        print(f"[FLASH] {rate} {port} verified", flush=True)
    time.sleep(1.0)


def run_cycle(rate, cycle, relay_args):
    cmd = [
        sys.executable, "-u", str(RELAY),
        "--tx-port", "COM23", "--rx-port", "COM4", "--baud", "115200",
        "--rate", rate, "--seconds", "2", "--chunk-ms", "500",
        "--play-gain", "4", "--beep", "--no-pcm-dump",
        "--inject-wav", str(WAV), "--buffer-play", "--timeout", "60",
        *relay_args,
    ]
    started = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=75,
        )
        output = result.stdout or ""
        timed_out = False
        rc = result.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        timed_out = True
        rc = -999

    frames = len(re.findall(r"^\[TOKEN\]", output, re.MULTILINE))
    relayed = len(re.findall(r"^\[RELAY\]", output, re.MULTILINE))
    played = 1 if re.search(r"played=1/1", output) else 0
    ok = not timed_out and rc == 0 and frames > 0 and relayed == 4 and played == 1
    record = {
        "rate": rate,
        "cycle": cycle,
        "ok": ok,
        "rc": rc,
        "timeout": timed_out,
        "elapsed_s": round(time.time() - started, 3),
        "token_frames": frames,
        "relayed_groups": relayed,
        "played": played,
    }
    log = OUT_DIR / rate / f"cycle_{cycle:02d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(output, encoding="utf-8")
    return record, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", nargs="+", default=["1k", "3k", "6k"])
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--restore", default="3k")
    parser.add_argument("--relay-arg", action="append", default=[])
    args = parser.parse_args()

    summary = {}
    for rate in args.rates:
        flash_rate(rate)
        records = []
        for cycle in range(1, args.cycles + 1):
            record, output = run_cycle(rate, cycle, args.relay_arg)
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if not record["ok"]:
                print(output[-2500:], flush=True)
                break
        summary[rate] = {
            "cycles_requested": args.cycles,
            "cycles_completed": len(records),
            "passed": sum(1 for item in records if item["ok"]),
            "failed": sum(1 for item in records if not item["ok"]),
            "records": records,
        }
        if any(not item["ok"] for item in records):
            break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = OUT_DIR / "summary.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[REPORT] {report}", flush=True)

    if args.restore:
        flash_rate(args.restore)
        rates.write_cached_rate(args.restore, rates.FIRMWARE_SHA256[args.restore])
    return 0 if all(item["failed"] == 0 for item in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
