#!/usr/bin/env python3
"""Create one role's application in a fresh directory; never edit the sources."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROFILE = REPO / "app/jixun-aibox"


def materialize(role, output):
    if role not in ("tx", "rx"):
        raise ValueError("role must be tx or rx")
    output = Path(output).resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("output must be outside the repository")
    if output.exists():
        raise FileExistsError("output already exists; choose a fresh directory")
    manifest = json.loads((PROFILE / "source_manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        actual = hashlib.sha256((PROFILE / name).read_bytes()).hexdigest()
        if actual != info["prepared_sha256"]:
            raise ValueError("source integrity mismatch: " + name)
    shutil.copytree(PROFILE / "common", output)
    if role == "rx":
        for source in (PROFILE / "roles/rx").rglob("*"):
            if source.is_file():
                target = output / source.relative_to(PROFILE / "roles/rx")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    role_header = (output / "port/jx_simple_role.h").read_text(encoding="utf-8")
    if "#define JX_SIMPLE_RX " + str(int(role == "rx")) not in role_header:
        raise ValueError("role header mismatch")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("tx", "rx"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(materialize(args.role, args.output))
