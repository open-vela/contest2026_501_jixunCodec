#!/usr/bin/env python3
"""Verify every stored float32 model tensor against the supplied mixed INT8 export.

This verifies the lossless storage adapter, not the numerical output of PIE kernels.
"""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

PROFILE = Path(__file__).resolve().parent.parent / "app/jixun-aibox"

def array(text, name):
    match = re.search(r"\b" + name + r"\[[^\]]*\][^=]*=\s*\{(.*?)\};", text, re.S)
    if match is None:
        raise ValueError("missing array: " + name)
    return [v.strip() for v in match[1].split(",") if v.strip()]

def number(value):
    value = value.rstrip("fF")
    return float.fromhex(value) if "0x" in value.lower() else float(value)

def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]

def table(text):
    return {name: [int(v) for v in re.findall(r"-?\d+", fields)]
            for name, fields in re.findall(r'\{"([^"\n]+)",([^\n]+)\}', text)}

def verify(reference, scope="decoder"):
    prepared = PROFILE / "common/weights/3k/weights_data.h"
    src = Path(reference).read_text(encoding="utf-8")
    dst = prepared.read_text(encoding="utf-8")
    a, b = table(src), table(dst)
    if set(a) != set(b):
        raise ValueError("tensor name mismatch")
    floats = [f32(number(v)) for v in array(src, "WT_FDATA")]
    ints = [int(v) for v in array(src, "WT_IDATA")]
    scales = [f32(number(v)) for v in array(src, "WT_SDATA")]
    values = [f32(number(v)) for v in array(dst, "WT_DATA")]
    count = 0
    compared = 0
    differences = []
    for name, row in a.items():
        off, n, nd, *tail = row
        shape, q, so, sn = tail[:4], *tail[4:]
        target = b[name]
        selected = scope == "all" or name.startswith("decoder.")
        if [n, nd] + shape != target[1:]:
            differences.append({"name":name,"kind":"shape",
                                "reference_shape":shape[:nd],"prepared_shape":target[3:3+target[2]]})
            if selected:
                raise ValueError("selected tensor shape mismatch: " + name)
            continue
        if q:
            if sn <= 0 or n % sn:
                raise ValueError("unsupported quantization layout: " + name)
            expected = [f32(ints[off+i] * scales[so+i//(n//sn)]) for i in range(n)]
        else:
            expected = floats[off:off+n]
        actual = values[target[0]:target[0]+n]
        equal = struct.pack("<"+"f"*n,*expected) == struct.pack("<"+"f"*n,*actual)
        if not equal:
            differences.append({"name":name,"kind":"values"})
            if selected:
                raise ValueError("selected tensor value mismatch: " + name)
        if selected:
            count += n
            compared += 1
    return {"scope":scope,"compared_tensors":compared,"compared_float32_values":count,
            "selected_float32_bit_identical":True,"whole_model_identical":not differences,
            "other_scope_differences":differences,
            "reference_sha256":hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
            "prepared_adapter_sha256":hashlib.sha256(prepared.read_bytes()).hexdigest(),
            "limitation":"dequantized storage adapter only; runtime kernels not evaluated"}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True,
                   help="weights_data_3k.h from the user-specified light-A INT8 model")
    p.add_argument("--scope", choices=("decoder", "all"), default="decoder")
    args = p.parse_args()
    print(json.dumps(verify(args.reference,args.scope),indent=2))
