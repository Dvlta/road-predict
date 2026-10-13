"""Pure-Python TFRecord reader/writer tests."""

import pytest

from prs.data.raw.tfrecord import TFRecordError, crc32c, read_records, write_records

RECORDS = [b"", b"a", bytes(range(256)) * 10]


def test_crc32c_known_vector():
    assert crc32c(b"123456789") == 0xE3069283


def test_roundtrip_with_crc(tmp_path):
    path = tmp_path / "x.tfrecord"
    write_records(path, RECORDS)
    assert list(read_records(path, verify_crc=True)) == RECORDS


def test_truncated_file_raises(tmp_path):
    path = tmp_path / "x.tfrecord"
    write_records(path, RECORDS)
    path.write_bytes(path.read_bytes()[:-3])
    with pytest.raises(TFRecordError, match="truncated"):
        list(read_records(path))


def test_corrupted_data_detected_only_with_crc(tmp_path):
    path = tmp_path / "x.tfrecord"
    write_records(path, [b"hello world"])
    data = bytearray(path.read_bytes())
    data[12] ^= 0xFF  # first payload byte
    path.write_bytes(bytes(data))
    assert list(read_records(path)) == [b"\x97ello world"]
    with pytest.raises(TFRecordError, match="data CRC"):
        list(read_records(path, verify_crc=True))
