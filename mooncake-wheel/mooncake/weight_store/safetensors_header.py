"""Read a safetensors file's total size from its header.

This is what lets the control plane work without a manifest: a safetensors
file is self-describing, so the reader can derive how large a shard is -- and
therefore how many chunks it occupies -- from its first bytes alone.

Layout::

    [8 bytes] header length N, little-endian u64
    [N bytes] header, UTF-8 JSON
    [...]     tensor data, at offsets relative to the end of the header

Every tensor entry carries ``data_offsets: [begin, end]`` relative to the end
of the header, so the file size is ``8 + N + max(end)``.

Deliberately NOT done here: probing chunk keys until one is missing. A node
failure can drop a chunk from the *middle* of a file, and probing would stop
at that hole and report a truncated file as complete. Sizes must come from
the header, and then every chunk key gets checked.
"""

from __future__ import annotations

import json
import struct

HEADER_LENGTH_SIZE = 8
# The header is metadata for one file; anything this large means the bytes are
# not a safetensors header (or are corrupt), and we should say so rather than
# try to parse a gigabyte of JSON.
MAX_HEADER_LENGTH = 256 * 1024 * 1024


class SafetensorsFormatError(ValueError):
    """The bytes are not a well-formed safetensors header."""


def header_length(prefix: bytes) -> int:
    """Read the header length from the first 8 bytes of a safetensors file."""
    if len(prefix) < HEADER_LENGTH_SIZE:
        raise SafetensorsFormatError(
            f"need {HEADER_LENGTH_SIZE} bytes to read the header length, "
            f"got {len(prefix)}"
        )
    (length,) = struct.unpack_from("<Q", prefix, 0)
    if length == 0:
        raise SafetensorsFormatError("header length is zero")
    if length > MAX_HEADER_LENGTH:
        raise SafetensorsFormatError(
            f"header length {length} exceeds the {MAX_HEADER_LENGTH} byte limit; "
            "this is probably not a safetensors file"
        )
    return length


def parse_header(header_bytes: bytes) -> dict:
    """Parse the JSON header of a safetensors file."""
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsFormatError(f"header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise SafetensorsFormatError(
            f"header must be a JSON object, got {type(header).__name__}"
        )
    return header


def data_end_offset(header: dict) -> int:
    """Largest tensor end offset in the header, relative to the data section."""
    end = 0
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise SafetensorsFormatError(f"tensor entry {name!r} is not an object")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
        ):
            raise SafetensorsFormatError(
                f"tensor entry {name!r} has invalid data_offsets: {offsets!r}"
            )
        begin, stop = offsets
        if begin < 0 or stop < begin:
            raise SafetensorsFormatError(
                f"tensor entry {name!r} has invalid data_offsets: {offsets!r}"
            )
        end = max(end, stop)
    return end


def total_size(prefix: bytes) -> int:
    """Total byte size of a safetensors file, from its leading bytes.

    ``prefix`` must cover the 8-byte length plus the whole JSON header; the
    first chunk of a chunked file is far larger than any real header.

    Raises SafetensorsFormatError if the prefix is too short to contain the
    full header, so a caller can fetch more rather than guess.
    """
    length = header_length(prefix)
    header_end = HEADER_LENGTH_SIZE + length
    if len(prefix) < header_end:
        raise SafetensorsFormatError(
            f"prefix of {len(prefix)} bytes does not cover the "
            f"{header_end} byte header"
        )
    header = parse_header(prefix[HEADER_LENGTH_SIZE:header_end])
    return header_end + data_end_offset(header)


def weight_map_files(index_payload: bytes) -> list[str]:
    """Shard filenames listed in a model.safetensors.index.json.

    Returned sorted and de-duplicated: the weight map is keyed by tensor name,
    so one shard appears once per tensor it holds.
    """
    try:
        index = json.loads(index_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsFormatError(
            f"safetensors index is not valid JSON: {exc}"
        ) from exc
    if not isinstance(index, dict):
        raise SafetensorsFormatError("safetensors index must be a JSON object")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SafetensorsFormatError("safetensors index has no usable weight_map")
    files = set()
    for tensor_name, filename in weight_map.items():
        if not isinstance(filename, str) or not filename:
            raise SafetensorsFormatError(
                f"weight_map entry {tensor_name!r} has invalid filename: "
                f"{filename!r}"
            )
        files.add(filename)
    return sorted(files)
