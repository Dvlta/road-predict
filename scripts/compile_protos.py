"""Regenerate the Python code for the vendored WOMD protos (design doc §28)."""

from __future__ import annotations

import sys
from pathlib import Path

import grpc_tools
from grpc_tools import protoc

SRC = Path(__file__).resolve().parents[1] / "src"
PROTOS = ["prs/data/raw/protos/map.proto", "prs/data/raw/protos/scenario.proto"]


def main() -> None:
    well_known = Path(grpc_tools.__file__).parent / "_proto"
    args = [
        "protoc",
        f"-I{SRC}",
        f"-I{well_known}",
        f"--python_out={SRC}",
        f"--pyi_out={SRC}",
        *PROTOS,
    ]
    if protoc.main(args) != 0:
        sys.exit("protoc failed")
    for proto in PROTOS:
        print(f"generated {SRC / proto.replace('.proto', '_pb2.py')}")


if __name__ == "__main__":
    main()
