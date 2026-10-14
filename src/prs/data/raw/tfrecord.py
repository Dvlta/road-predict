"""Pure-Python TFRecord reading and writing (design doc §28). No TensorFlow.

Record layout:
    uint64  length                      (little-endian)
    uint32  masked CRC-32C of length
    bytes   data[length]
    uint32  masked CRC-32C of data
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Iterator
from pathlib import Path

_HEADER = struct.Struct("<QI")
_FOOTER = struct.Struct("<I")
_MASK_DELTA = 0xA282EAD8


class TFRecordError(ValueError):
    """A truncated or corrupted TFRecord file."""


def _crc32c_table() -> list[int]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = ((crc >> 1) ^ 0x82F63B78) if crc & 1 else crc >> 1
        table.append(crc)
    return table


_CRC32C_TABLE = _crc32c_table()


def crc32c(data: bytes) -> int:
    """CRC-32C (Castagnoli). Pure Python: fine for tests and spot checks, slow for whole shards."""
    crc = 0xFFFFFFFF
    table = _CRC32C_TABLE
    for byte in data:
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    crc = crc32c(data)
    rotated = ((crc >> 15) | (crc << 17)) & 0xFFFFFFFF
    return (rotated + _MASK_DELTA) & 0xFFFFFFFF


def read_records(path: str | Path, verify_crc: bool = False) -> Iterator[bytes]:
    """Yield each record's payload. CRC checks are off by default (downloads are MD5-verified)."""
    path = Path(path)
    with path.open("rb") as f:
        offset = 0
        while header := f.read(_HEADER.size):
            if len(header) < _HEADER.size:
                raise TFRecordError(f"{path}: truncated record header at byte {offset}")
            length, length_crc = _HEADER.unpack(header)
            data = f.read(length)
            footer = f.read(_FOOTER.size)
            if len(data) < length or len(footer) < _FOOTER.size:
                raise TFRecordError(f"{path}: truncated record at byte {offset}")
            if verify_crc:
                if masked_crc32c(header[:8]) != length_crc:
                    raise TFRecordError(f"{path}: length CRC mismatch at byte {offset}")
                if masked_crc32c(data) != _FOOTER.unpack(footer)[0]:
                    raise TFRecordError(f"{path}: data CRC mismatch at byte {offset}")
            yield data
            offset += _HEADER.size + length + _FOOTER.size


def write_records(path: str | Path, records: Iterable[bytes]) -> None:
    with Path(path).open("wb") as f:
        for data in records:
            length = struct.pack("<Q", len(data))
            f.write(length)
            f.write(_FOOTER.pack(masked_crc32c(length)))
            f.write(data)
            f.write(_FOOTER.pack(masked_crc32c(data)))
