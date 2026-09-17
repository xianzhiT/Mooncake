"""Deterministic key layout for weight files in Mooncake Store.

Every key is derived from the checkpoint id plus the file's path inside the
model directory, so a reader that knows what a model should contain can
compute the keys itself. Nothing in the store has to record the layout.

Two shapes, decided by whether the file is a weight shard:

    weight/models/<ckpt>/files/<path>                     whole object
    weight/models/<ckpt>/files/<path>/chunks/<00000000>   chunked

Small files (config.json, tokenizer.json, the safetensors index) are stored
whole, so one get retrieves them without knowing their size first. Weight
shards are chunked so a large file can be transferred in parallel.
"""

from __future__ import annotations

import re

CHUNK_SEGMENT = "chunks"
MODELS_PREFIX = "weight/models"
CHUNK_INDEX_WIDTH = 8

# Files that get chunked. Everything else is stored whole.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")

# HuggingFace fixed names. A reader can compute these keys directly.
CONFIG_FILE = "config.json"
SAFETENSORS_INDEX_FILE = "model.safetensors.index.json"

_CHECKPOINT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MAX_CHECKPOINT_ID_LEN = 255


def validate_checkpoint_id(checkpoint_id: str) -> None:
    if not isinstance(checkpoint_id, str) or not (
        1 <= len(checkpoint_id) <= _MAX_CHECKPOINT_ID_LEN
    ):
        raise ValueError(f"invalid checkpoint_id: {checkpoint_id!r}")
    if not _CHECKPOINT_ID_RE.fullmatch(checkpoint_id):
        raise ValueError(f"invalid checkpoint_id: {checkpoint_id!r}")


def validate_relative_path(relative_path: str) -> None:
    """Reject paths that would break the key layout or escape the model dir.

    The path becomes part of the key verbatim, so it must not be absolute,
    contain traversal segments, or collide with the reserved chunk segment.
    """
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"invalid relative_path: {relative_path!r}")
    if relative_path.startswith("/") or relative_path.endswith("/"):
        raise ValueError(f"invalid relative_path: {relative_path!r}")
    segments = relative_path.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise ValueError(f"invalid relative_path: {relative_path!r}")
    if CHUNK_SEGMENT in segments:
        # Otherwise a file literally named "chunks" could alias a chunk key.
        raise ValueError(
            f"relative_path must not contain a {CHUNK_SEGMENT!r} segment: "
            f"{relative_path!r}"
        )


def is_weight_file(relative_path: str) -> bool:
    """Whether this file is chunked (see WEIGHT_SUFFIXES)."""
    return relative_path.lower().endswith(WEIGHT_SUFFIXES)


def model_prefix(checkpoint_id: str) -> str:
    """Prefix covering every key of one checkpoint, for listing or deleting."""
    validate_checkpoint_id(checkpoint_id)
    return f"{MODELS_PREFIX}/{checkpoint_id}/"


def model_file_key(checkpoint_id: str, relative_path: str) -> str:
    """Key of a whole-object file."""
    validate_checkpoint_id(checkpoint_id)
    validate_relative_path(relative_path)
    return f"{MODELS_PREFIX}/{checkpoint_id}/files/{relative_path}"


def model_file_chunk_key(
    checkpoint_id: str, relative_path: str, chunk_index: int
) -> str:
    """Key of one chunk of a chunked file."""
    if not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError(f"invalid chunk_index: {chunk_index!r}")
    base = model_file_key(checkpoint_id, relative_path)
    return f"{base}/{CHUNK_SEGMENT}/{chunk_index:0{CHUNK_INDEX_WIDTH}d}"


def model_file_chunk_keys(
    checkpoint_id: str, relative_path: str, chunk_count: int
) -> list[str]:
    """Keys of all chunks of a file, for a known chunk count."""
    if not isinstance(chunk_count, int) or chunk_count < 0:
        raise ValueError(f"invalid chunk_count: {chunk_count!r}")
    return [
        model_file_chunk_key(checkpoint_id, relative_path, index)
        for index in range(chunk_count)
    ]


def model_config_key(checkpoint_id: str) -> str:
    """Key of config.json, which every HuggingFace model has exactly one of."""
    return model_file_key(checkpoint_id, CONFIG_FILE)


def model_safetensors_index_key(checkpoint_id: str) -> str:
    """Key of the safetensors index. Absent for single-file models."""
    return model_file_key(checkpoint_id, SAFETENSORS_INDEX_FILE)


def config_listing_pattern() -> str:
    """Regex matching exactly one key per stored model.

    Used with the store's regex key query to enumerate models. config.json is
    the marker because every HuggingFace model has exactly one (the
    safetensors index is absent for single-file models), and it is stored
    whole, so the pattern cannot also match chunk keys.
    """
    return rf"^{MODELS_PREFIX}/[^/]+/files/{re.escape(CONFIG_FILE)}$"


def model_prefix_pattern(checkpoint_id: str) -> str:
    """Regex matching every key of one checkpoint."""
    return f"^{re.escape(model_prefix(checkpoint_id))}"


def checkpoint_id_from_config_key(key: str) -> str:
    """Recover the checkpoint id from a config.json key.

    Inverse of model_config_key, used to turn listing results into model ids.
    """
    match = re.fullmatch(
        rf"{MODELS_PREFIX}/([^/]+)/files/{re.escape(CONFIG_FILE)}", key
    )
    if match is None:
        raise ValueError(f"not a config key: {key!r}")
    return match.group(1)


def chunk_count_for_size(size: int, chunk_size: int) -> int:
    """How many chunks a file of this size occupies."""
    if size < 0:
        raise ValueError(f"invalid size: {size!r}")
    if chunk_size <= 0:
        raise ValueError(f"invalid chunk_size: {chunk_size!r}")
    if size == 0:
        # An empty file still occupies one (empty) chunk, so that its presence
        # is observable and it round-trips through put/get like any other.
        return 1
    return (size + chunk_size - 1) // chunk_size
