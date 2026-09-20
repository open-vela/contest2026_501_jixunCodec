#!/usr/bin/env python3
"""Validate an already-flashed Jixun Codec benchmark over serial."""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time

import serial


KNOWN = {
    "1k": "70cc287327d70e4b:3aa0e5aa76f02e8c",
    "3k": "73e73efd4dbdc601:10cf457b03bfcf6e",
    "6k": "6466db7b1b12fa53:069f2b9cf1375390",
}
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
RATE_RE = re.compile(r"码率\s+([136]k)")
ENCODE_RE = re.compile(r"编码耗时\s*:\s*([0-9.]+)\s*ms")
DECODE_RE = re.compile(r"解码耗时\s*:\s*([0-9.]+)\s*ms")
TOTAL_RE = re.compile(r"单次总耗时\s*:\s*([0-9.]+)\s*ms")
FP_RE = re.compile(r"指纹\s*:\s*idx=([0-9a-fA-F]+)\s+rec=([0-9a-fA-F]+)")


def clean(data: bytes) -> str:
    return ANSI_RE.sub("", data.decode("utf-8", "replace"))


def wait_prompt(port: serial.Serial, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    data = bytearray()
    while time.monotonic() < deadline:
        chunk = port.read(4096)
        if chunk:
            data.extend(chunk)
            if b"nsh>" in data:
                return clean(bytes(data))
    raise TimeoutError(f"NSH prompt timeout after {timeout:.0f}s")


def command(port: serial.Serial, text: str, timeout: float) -> str:
    port.reset_input_buffer()
    port.write(text.encode("ascii") + b"\r")
    port.flush()
    return wait_prompt(port, timeout)


def one_decimal(value: float | None) -> float | None:
    return round(value, 1) if value is not None else None


def parse(output: str) -> dict:
    rate = RATE_RE.search(output)
    enc = ENCODE_RE.search(output)
    dec = DECODE_RE.search(output)
    total = TOTAL_RE.search(output)
    fp = FP_RE.search(output)
    fingerprint = f"{fp.group(1).lower()}:{fp.group(2).lower()}" if fp else None
    return {
        "rate": rate.group(1) if rate else None,
        "encode_ms": one_decimal(float(enc.group(1))) if enc else None,
        "decode_ms": one_decimal(float(dec.group(1))) if dec else None,
        "total_ms": one_decimal(float(total.group(1))) if total else None,
        "fingerprint": fingerprint,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--expect-rate", choices=tuple(KNOWN))
    parser.add_argument("--expect-fingerprint")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    port = serial.Serial(args.port, 115200, timeout=0.1, write_timeout=5.0)
    port.dtr = False
    port.rts = False
    rows = []
    try:
        port.write(b"\r\n")
        port.flush()
        wait_prompt(port, 90.0)
        for key in ("JX_USB", "JX_PCM_DUMP", "JX_BEEP"):
            command(port, f"set {key} 0", 10.0)
        for round_no in range(1, args.rounds + 1):
            output = command(port, "jixun bench 1", args.timeout)
            row = parse(output)
            row["round"] = round_no
            row["assertion"] = "Assertion failed" in output or "assertion failed" in output
            rows.append(row)
            print(
                f"[{row['rate']}] round={round_no} "
                f"enc={row['encode_ms']}ms dec={row['decode_ms']}ms "
                f"total={row['total_ms']}ms fp={row['fingerprint']}",
                flush=True,
            )
            if row["assertion"] or row["fingerprint"] is None or row["total_ms"] is None:
                raise RuntimeError(f"round {round_no} produced no valid benchmark result")
    finally:
        port.close()

    rates = {row["rate"] for row in rows}
    fingerprints = {row["fingerprint"] for row in rows}
    if len(rates) != 1:
        raise RuntimeError(f"inconsistent reported rates: {sorted(rates)}")
    rate = rates.pop()
    if args.expect_rate and rate != args.expect_rate:
        raise RuntimeError(f"expected rate {args.expect_rate}, got {rate}")
    expected = args.expect_fingerprint or KNOWN.get(rate)
    if expected and fingerprints != {expected}:
        raise RuntimeError(f"fingerprint mismatch: expected {expected}, got {sorted(fingerprints)}")

    totals = [row["total_ms"] for row in rows]
    print(
        f"RESULT rate={rate} rounds={len(rows)} "
        f"min={min(totals):.1f} mean={statistics.fmean(totals):.1f} "
        f"max={max(totals):.1f} stdev={statistics.stdev(totals) if len(totals) > 1 else 0:.1f} "
        f"fingerprint={next(iter(fingerprints))}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
