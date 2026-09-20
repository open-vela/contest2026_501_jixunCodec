#!/usr/bin/env python3
"""Run deterministic codec benchmark cycles on one or both ESP32 boards.

The flasher only writes the application slot at 0x10000. Each benchmark
round is bounded by the next NSH prompt and saved verbatim for audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import serial


ROOT = Path(__file__).resolve().parent
FLASH_DIR = ROOT / "firmware"
OUT_ROOT = ROOT / "_pc_voice_rx" / "cycle_50"

RATES = {
    "1k": {
        "image": FLASH_DIR / "nuttx_r276_lcdinit_1k.bin",
        "sha256": "d4034ea120339b6c325bba0574412b3d8fc1cb8cc606626253774dd24e5ba3cc",
    },
    "3k": {
        "image": FLASH_DIR / "nuttx_r276_lcdinit_3k.bin",
        "sha256": "be321d5f1685e4064ac2ba4c6fd3e461bd25e3a53287c7e6cd8a5dad4e5db93b",
    },
    "6k": {
        "image": FLASH_DIR / "nuttx_r276_lcdinit_6k.bin",
        "sha256": "37b02371f51984e3af8eb16ed841be2b5923c9ad53d297297214cdc2b9bf6352",
    },
}

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
ENCODE_RE = re.compile(r"编码耗时\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")
DECODE_RE = re.compile(r"解码耗时\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")
TOTAL_RE = re.compile(r"单次总耗时\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")
FINGERPRINT_RE = re.compile(r"指纹\s*:\s*idx=([0-9a-fA-F]+)\s+rec=([0-9a-fA-F]+)")
RATE_RE = re.compile(r"码率\s+([136]k)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(data: bytes) -> str:
    return ANSI_RE.sub("", data.decode("utf-8", "replace")).replace("\x00", "")


def wait_for_prompt(port: serial.Serial, timeout_s: float) -> bytes:
    deadline = time.monotonic() + timeout_s
    data = bytearray()
    while time.monotonic() < deadline:
        chunk = port.read(4096)
        if chunk:
            data.extend(chunk)
            if b"nsh>" in data:
                return bytes(data)
    raise TimeoutError(f"NSH prompt timeout after {timeout_s:.0f}s")


def run_command(port: serial.Serial, command: str, timeout_s: float) -> str:
    port.reset_input_buffer()
    port.write(command.encode("ascii") + b"\r")
    port.flush()
    return clean_text(wait_for_prompt(port, timeout_s))


def flash_board(port_name: str, rate: str, image: Path, digest: str, out_dir: Path) -> None:
    cmd = [
        sys.executable, "-m", "esptool", "--port", port_name,
        "--baud", "921600", "--after", "hard-reset",
        "write-flash", "0x10000", str(image),
    ]
    print(f"[FLASH] {rate} -> {port_name} sha256={digest}", flush=True)
    result = subprocess.run(
        cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    log_path = out_dir / f"flash_{port_name}.log"
    log_path.write_text(result.stdout or "", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"flash failed on {port_name}; see {log_path}")
    if "Hash of data verified." not in (result.stdout or ""):
        raise RuntimeError(f"flash verification marker missing on {port_name}")
    if sha256(image) != digest:
        raise RuntimeError(f"firmware changed while flashing {image}")
    print(f"[FLASH] {rate} -> {port_name} verified", flush=True)


def parse_round(text: str) -> dict:
    enc = ENCODE_RE.search(text)
    dec = DECODE_RE.search(text)
    total = TOTAL_RE.search(text)
    fp = FINGERPRINT_RE.search(text)
    rate = RATE_RE.search(text)
    result = {
        "encode_ms": float(enc.group(1)) if enc else None,
        "decode_ms": float(dec.group(1)) if dec else None,
        "total_ms": float(total.group(1)) if total else None,
        "idx": fp.group(1).lower() if fp else None,
        "rec": fp.group(2).lower() if fp else None,
        "reported_rate": rate.group(1) if rate else None,
        "assertion": "Assertion failed" in text or "assertion failed" in text,
    }
    result["fingerprint"] = (
        f"{result['idx']}:{result['rec']}" if result["idx"] and result["rec"] else None
    )
    result["ok"] = (
        not result["assertion"]
        and all(result[key] is not None for key in ("encode_ms", "decode_ms", "total_ms"))
        and result["fingerprint"] is not None
        and result["reported_rate"] is not None
    )
    return result


def stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values), "min": min(values), "mean": statistics.fmean(values),
        "median": statistics.median(values), "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def run_board(
    port_name: str, rate: str, cycles: int, timeout_s: float, out_dir: Path
) -> dict:
    port = serial.Serial(port_name, 115200, timeout=0.1, write_timeout=5.0)
    port.dtr = False
    port.rts = False
    rows: list[dict] = []
    try:
        port.write(b"\r\n")
        port.flush()
        print(f"[BOOT] {port_name} waiting for NSH", flush=True)
        wait_for_prompt(port, 90.0)
        print(f"[BOOT] {port_name} ready", flush=True)
        for key, value in (("JX_USB", "0"), ("JX_PCM_DUMP", "0"), ("JX_BEEP", "0")):
            output = run_command(port, f"set {key} {value}", 10.0)
            if f"set {key} {value}" not in output:
                raise RuntimeError(f"{port_name}: no echo for set {key}")
        for cycle in range(1, cycles + 1):
            started = time.monotonic()
            try:
                raw = run_command(port, "jixun bench 1", timeout_s)
                error = None
            except TimeoutError as exc:
                raw = ""
                error = str(exc)
            elapsed_s = time.monotonic() - started
            row = {
                "rate": rate, "port": port_name, "cycle": cycle,
                "elapsed_s": round(elapsed_s, 3), **parse_round(raw), "error": error,
            }
            if row["reported_rate"] != rate:
                row["ok"] = False
                row["error"] = row["error"] or (
                    f"reported rate {row['reported_rate']!r} != {rate!r}"
                )
            rows.append(row)
            (out_dir / "raw" / f"cycle_{cycle:03d}.log").write_text(raw, encoding="utf-8")
            print(
                f"[{rate} {port_name}] {cycle:02d}/{cycles} "
                f"{'OK' if row['ok'] else 'FAIL'} enc={row['encode_ms']} "
                f"dec={row['decode_ms']} total={row['total_ms']} "
                f"wall={elapsed_s:.2f}s fp={row['fingerprint']}", flush=True,
            )
            if not row["ok"]:
                raise RuntimeError(
                    f"{port_name} cycle {cycle} failed: {row['error'] or 'parse error'}"
                )
    finally:
        port.close()
    totals = [float(row["total_ms"]) for row in rows if row["total_ms"] is not None]
    encodes = [float(row["encode_ms"]) for row in rows if row["encode_ms"] is not None]
    decodes = [float(row["decode_ms"]) for row in rows if row["decode_ms"] is not None]
    fingerprints = [row["fingerprint"] for row in rows if row["fingerprint"]]
    return {
        "rate": rate, "port": port_name, "cycles_requested": cycles,
        "cycles_completed": len(rows), "failures": sum(not row["ok"] for row in rows),
        "unique_fingerprints": sorted(set(fingerprints)),
        "consistent": len(fingerprints) == cycles and len(set(fingerprints)) == 1,
        "encode_ms": stats(encodes), "decode_ms": stats(decodes),
        "total_ms": stats(totals), "rows": rows,
    }


def write_reports(rate: str, summaries: list[dict], out_dir: Path) -> None:
    fieldnames = [
        "rate", "port", "cycle", "encode_ms", "decode_ms", "total_ms",
        "elapsed_s", "idx", "rec", "fingerprint", "reported_rate",
        "ok", "assertion", "error",
    ]
    all_rows = [row for summary in summaries for row in summary["rows"]]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    report = {
        "rate": rate, "firmware_sha256": RATES[rate]["sha256"],
        "firmware": str(RATES[rate]["image"]),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "summaries": summaries,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[REPORT] {out_dir / 'summary.json'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", choices=tuple(RATES), required=True)
    parser.add_argument("--ports", nargs="+", default=["COM4", "COM23"])
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-flash", action="store_true")
    args = parser.parse_args()
    config = RATES[args.rate]
    image = config["image"]
    digest = config["sha256"]
    if not image.exists():
        raise SystemExit(f"missing firmware: {image}")
    if sha256(image) != digest:
        raise SystemExit(f"firmware hash mismatch: {image}")
    out_dir = OUT_ROOT / args.rate
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    summaries = []
    for port_name in args.ports:
        board_dir = out_dir / f"port_{port_name}"
        board_dir.mkdir(parents=True, exist_ok=True)
        (board_dir / "raw").mkdir(parents=True, exist_ok=True)
        if not args.no_flash:
            flash_board(port_name, args.rate, image, digest, board_dir)
        summary = run_board(port_name, args.rate, args.cycles, args.timeout, board_dir)
        summaries.append(summary)
        (board_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_reports(args.rate, summaries, out_dir)
    print(
        f"[DONE] rate={args.rate} boards={len(summaries)} "
        f"cycles={sum(item['cycles_completed'] for item in summaries)}", flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
