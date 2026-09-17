"""Manifest-less weight file cache on top of Mooncake Store.

The store holds bytes only. It records nothing about what a model is, and
there is no manifest, no status field, no registry of model names, and no
lock. Everything a reader needs is either computable from the checkpoint id
(see model_keyspace) or carried by the model's own files: config.json, the
safetensors index, and each shard's self-describing header.

Consequences worth stating plainly:

* Completeness is decided by checking the keys that should exist, right now.
  Nothing claims a model is ready, so nothing can claim it while its chunks
  are gone -- which happens routinely, since the store is volatile memory and
  a restarted node loses its chunks.
* Concurrent imports of the same checkpoint write identical bytes to
  identical keys, so they cannot corrupt each other, and a reader mid-import
  simply sees an incomplete model.
* Re-importing a *changed* checkpoint under the same id is refused, because
  the store's put is write-once and would silently keep the old bytes. Delete
  first, or pass force=True.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from . import safetensors_header
from .model_keyspace import (
    CONFIG_FILE,
    SAFETENSORS_INDEX_FILE,
    checkpoint_id_from_config_key,
    chunk_count_for_size,
    config_listing_pattern,
    is_weight_file,
    model_config_key,
    model_file_chunk_key,
    model_file_chunk_keys,
    model_file_key,
    model_prefix_pattern,
    model_safetensors_index_key,
    validate_checkpoint_id,
    validate_relative_path,
)

DEFAULT_FILE_CHUNK_SIZE = 64 * 1024 * 1024
# store.remove returns this for an object that is already gone; tolerated so
# that re-running a partial delete stays idempotent.
MISSING_OBJECT_ERROR = -704


class WeightStoreError(RuntimeError):
    """Base class for control-plane failures."""


class CheckpointExistsError(WeightStoreError):
    """Keys already exist under this checkpoint id.

    Raised instead of importing over them, because put is write-once: the new
    bytes would be discarded and every caller would still be told the import
    succeeded.
    """


class CheckpointNotFoundError(WeightStoreError):
    """Nothing is stored under this checkpoint id."""


class IncompleteCheckpointError(WeightStoreError):
    """Some of the checkpoint's chunks are missing from the store."""

    def __init__(self, checkpoint_id: str, missing_keys: List[str]) -> None:
        self.checkpoint_id = checkpoint_id
        self.missing_keys = missing_keys
        shown = ", ".join(missing_keys[:3])
        if len(missing_keys) > 3:
            shown += f", ... (+{len(missing_keys) - 3} more)"
        super().__init__(
            f"checkpoint {checkpoint_id!r} is missing "
            f"{len(missing_keys)} object(s): {shown}"
        )


@dataclass(frozen=True)
class FileLayout:
    """Where one file's bytes live, derived rather than recorded."""

    path: str
    size: int
    chunked: bool
    keys: List[str]


@dataclass
class CheckpointStatus:
    """What the store actually holds for a checkpoint, right now."""

    checkpoint_id: str
    files: List[FileLayout] = field(default_factory=list)
    missing_keys: List[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing_keys

    @property
    def total_size(self) -> int:
        return sum(item.size for item in self.files)

    def to_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "complete": self.complete,
            "total_size": self.total_size,
            "files": [
                {
                    "path": item.path,
                    "size": item.size,
                    "chunked": item.chunked,
                    "chunk_count": len(item.keys),
                }
                for item in self.files
            ],
            "missing_keys": self.missing_keys,
        }


class WeightCacheClient:
    """Import, inspect, and delete model files in Mooncake Store."""

    def __init__(
        self,
        store,
        *,
        replica_num: int = 1,
        file_chunk_size: int = DEFAULT_FILE_CHUNK_SIZE,
        hard_pin_weights: bool = True,
        progress: bool = True,
    ) -> None:
        if file_chunk_size <= 0:
            raise ValueError(f"invalid file_chunk_size: {file_chunk_size!r}")
        self.store = store
        self.replica_num = replica_num
        self.file_chunk_size = file_chunk_size
        self.hard_pin_weights = hard_pin_weights
        self.progress = progress

    # ---------------------------------------------------------------- import

    def import_model(
        self,
        checkpoint_id: str,
        source: str | os.PathLike[str],
        *,
        force: bool = False,
    ) -> CheckpointStatus:
        """Mirror a model directory into the store.

        Files are copied as they are: weight shards chunked, everything else
        whole. No index or manifest is synthesised -- a single-file model that
        has no safetensors index is stored without one, exactly as on disk.
        """
        validate_checkpoint_id(checkpoint_id)
        source_dir = Path(source)
        if not source_dir.is_dir():
            raise WeightStoreError(f"source is not a directory: {source_dir}")

        relative_paths = _list_model_files(source_dir)
        if not relative_paths:
            raise WeightStoreError(f"no files found under {source_dir}")
        if CONFIG_FILE not in relative_paths:
            # Without it a reader cannot bootstrap, and `list` cannot see the
            # model at all, since config.json is the listing marker.
            raise WeightStoreError(
                f"{source_dir} has no {CONFIG_FILE}; "
                "it does not look like a HuggingFace model directory"
            )

        existing = self.store.query_keys_by_regex(model_prefix_pattern(checkpoint_id))
        if existing:
            if not force:
                raise CheckpointExistsError(
                    f"checkpoint {checkpoint_id!r} already has "
                    f"{len(existing)} object(s) in the store. Delete it first, "
                    "or pass force=True: put is write-once, so importing over "
                    "them would keep the old bytes and still report success."
                )
            self._log(f"force: clearing {len(existing)} existing object(s)")
            self.delete_model(checkpoint_id)

        written: List[str] = []
        try:
            for index, relative_path in enumerate(relative_paths, start=1):
                layout = self._put_file(
                    checkpoint_id,
                    source_dir / relative_path,
                    relative_path,
                    written,
                )
                self._log(
                    f"[{index}/{len(relative_paths)}] {relative_path} "
                    f"({_format_bytes(layout.size)}, "
                    f"{len(layout.keys)} object(s))"
                )
        except BaseException:
            # Leftover chunks would not make an incomplete model look ready --
            # readers check every key -- but they would waste memory and block
            # a later import, so clear them.
            self._log("import failed; removing objects written so far")
            self._remove_keys(written)
            raise

        return self.inspect_model(checkpoint_id)

    # ------------------------------------------------------------- inspect

    def list_models(self) -> List[str]:
        """Checkpoint ids present in the store.

        Matches one well-known key per model (config.json) rather than
        enumerating every chunk, so the response stays small. O(total keys) on
        the master: intended for a management command, not a hot path.
        """
        keys = self.store.query_keys_by_regex(config_listing_pattern())
        return sorted(checkpoint_id_from_config_key(key) for key in keys)

    def inspect_model(self, checkpoint_id: str) -> CheckpointStatus:
        """Report what the store holds for a checkpoint, right now.

        Derives the expected key set from the model's own files, then checks
        every key. A model whose chunks were lost with a restarted node is
        reported incomplete, with the missing keys named.
        """
        validate_checkpoint_id(checkpoint_id)
        layouts = self._derive_layouts(checkpoint_id)
        expected: List[str] = []
        for layout in layouts:
            expected.extend(layout.keys)
        missing = self._missing_keys(expected)
        return CheckpointStatus(
            checkpoint_id=checkpoint_id,
            files=layouts,
            missing_keys=missing,
        )

    def verify_model(self, checkpoint_id: str) -> CheckpointStatus:
        """Read every byte back and confirm each shard parses.

        Stronger than inspect_model, which only checks that keys exist. No
        stored checksum is compared, because none is stored; instead each
        safetensors shard is re-parsed and its declared size checked against
        the bytes actually present, which catches truncation and corruption of
        the header region.
        """
        status = self.inspect_model(checkpoint_id)
        if not status.complete:
            raise IncompleteCheckpointError(checkpoint_id, status.missing_keys)

        for layout in status.files:
            payload = self._read_keys(layout.keys)
            if len(payload) != layout.size:
                raise WeightStoreError(
                    f"{layout.path}: expected {layout.size} bytes, "
                    f"read {len(payload)}"
                )
            if layout.path.endswith(".safetensors"):
                declared = safetensors_header.total_size(payload)
                if declared != len(payload):
                    raise WeightStoreError(
                        f"{layout.path}: header declares {declared} bytes, "
                        f"store holds {len(payload)}"
                    )
            self._log(f"verified {layout.path} ({_format_bytes(layout.size)})")
        return status

    # -------------------------------------------------------------- delete

    def delete_model(self, checkpoint_id: str) -> int:
        """Remove every object of a checkpoint. Idempotent.

        Matches by prefix rather than reading a record of what was written, so
        it also sweeps chunks left behind by an interrupted import.

        Passes force, because a read grants the object a lease (10s by
        default) and an unforced removal skips still-leased keys. Without it,
        deleting a checkpoint that was just inspected or loaded would silently
        leave most of it behind. Deletion here is an explicit operator action,
        not eviction, so a reader's lease must not veto it.
        """
        validate_checkpoint_id(checkpoint_id)
        removed = self._remove_by_regex(model_prefix_pattern(checkpoint_id))
        if removed < 0:
            raise WeightStoreError(
                f"failed to delete checkpoint {checkpoint_id!r}: {removed}"
            )
        self._log(f"removed {removed} object(s)")
        return int(removed)

    # ---------------------------------------------------------- materialize

    def materialize_file(
        self,
        checkpoint_id: str,
        relative_path: str,
        output_path: str | os.PathLike[str],
    ) -> int:
        """Write one stored file out to disk.

        Used for the small files (config, tokenizer) that SGLang's loaders
        need as real paths. Weight shards are streamed into registered memory
        instead and never land on disk.
        """
        validate_checkpoint_id(checkpoint_id)
        validate_relative_path(relative_path)
        size = self._file_size(checkpoint_id, relative_path)
        if size is None:
            raise CheckpointNotFoundError(
                f"{checkpoint_id}/{relative_path} is not in the store"
            )
        layout = self._layout_for(checkpoint_id, relative_path, size)
        payload = self._read_keys(layout.keys)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return len(payload)

    def read_file(self, checkpoint_id: str, relative_path: str) -> bytes:
        """Read one stored file into memory."""
        validate_checkpoint_id(checkpoint_id)
        validate_relative_path(relative_path)
        size = self._file_size(checkpoint_id, relative_path)
        if size is None:
            raise CheckpointNotFoundError(
                f"{checkpoint_id}/{relative_path} is not in the store"
            )
        layout = self._layout_for(checkpoint_id, relative_path, size)
        return self._read_keys(layout.keys)

    # ------------------------------------------------------------ internals

    def _derive_layouts(self, checkpoint_id: str) -> List[FileLayout]:
        """Work out which files a checkpoint should have, and their keys.

        Reads the model's own metadata rather than a stored description:
        config.json proves the model is there, the safetensors index names the
        shards, and each shard's header gives its size.
        """
        config = self._get_optional(model_config_key(checkpoint_id))
        if config is None:
            raise CheckpointNotFoundError(
                f"checkpoint {checkpoint_id!r} is not in the store "
                f"(no {CONFIG_FILE})"
            )

        layouts = [
            FileLayout(
                path=CONFIG_FILE,
                size=len(config),
                chunked=False,
                keys=[model_config_key(checkpoint_id)],
            )
        ]

        index_payload = self._get_optional(model_safetensors_index_key(checkpoint_id))
        if index_payload is None:
            # Single-file model: no index was stored, so look for the one
            # conventional shard name.
            shard_names = ["model.safetensors"]
        else:
            layouts.append(
                FileLayout(
                    path=SAFETENSORS_INDEX_FILE,
                    size=len(index_payload),
                    chunked=False,
                    keys=[model_safetensors_index_key(checkpoint_id)],
                )
            )
            shard_names = safetensors_header.weight_map_files(index_payload)

        for shard in shard_names:
            size = self._weight_shard_size(checkpoint_id, shard)
            if size is None:
                # Name the first chunk as the missing key, so the caller sees
                # which shard is gone rather than a bare "not found".
                layouts.append(
                    FileLayout(
                        path=shard,
                        size=0,
                        chunked=True,
                        keys=[model_file_chunk_key(checkpoint_id, shard, 0)],
                    )
                )
                continue
            layouts.append(self._layout_for(checkpoint_id, shard, size))
        return layouts

    def _layout_for(
        self, checkpoint_id: str, relative_path: str, size: int
    ) -> FileLayout:
        if is_weight_file(relative_path):
            count = chunk_count_for_size(size, self.file_chunk_size)
            return FileLayout(
                path=relative_path,
                size=size,
                chunked=True,
                keys=model_file_chunk_keys(checkpoint_id, relative_path, count),
            )
        return FileLayout(
            path=relative_path,
            size=size,
            chunked=False,
            keys=[model_file_key(checkpoint_id, relative_path)],
        )

    def _weight_shard_size(
        self, checkpoint_id: str, relative_path: str
    ) -> Optional[int]:
        """Size of a stored shard, read from its own header.

        Fetches only the first chunk: a safetensors header is far smaller than
        one chunk, so the total size is derivable without reading the shard.
        """
        first_chunk = self._get_optional(
            model_file_chunk_key(checkpoint_id, relative_path, 0)
        )
        if first_chunk is None:
            return None
        if not relative_path.endswith(".safetensors"):
            # Other chunked formats are not self-describing, so the count can
            # only come from the chunks present. A hole would be invisible
            # here; safetensors is the supported path.
            return self._probe_chunked_size(checkpoint_id, relative_path)
        return safetensors_header.total_size(first_chunk)

    def _probe_chunked_size(self, checkpoint_id: str, relative_path: str) -> int:
        size = 0
        index = 0
        while True:
            chunk = self._get_optional(
                model_file_chunk_key(checkpoint_id, relative_path, index)
            )
            if chunk is None:
                return size
            size += len(chunk)
            index += 1

    def _file_size(self, checkpoint_id: str, relative_path: str) -> Optional[int]:
        if is_weight_file(relative_path):
            return self._weight_shard_size(checkpoint_id, relative_path)
        payload = self._get_optional(model_file_key(checkpoint_id, relative_path))
        return None if payload is None else len(payload)

    def _put_file(
        self,
        checkpoint_id: str,
        source_path: Path,
        relative_path: str,
        written: List[str],
    ) -> FileLayout:
        """Store one file, appending each key to ``written`` as it lands.

        ``written`` is updated per key rather than per file so that a failure
        partway through a large shard still leaves the caller able to clean up
        the chunks that did land.
        """
        size = source_path.stat().st_size
        if not is_weight_file(relative_path):
            payload = source_path.read_bytes()
            key = model_file_key(checkpoint_id, relative_path)
            self._put(key, payload, self._config(chunked=False))
            written.append(key)
            return FileLayout(path=relative_path, size=size, chunked=False, keys=[key])

        config = self._config(chunked=True)
        keys: List[str] = []
        total = 0
        with source_path.open("rb") as handle:
            for index in range(chunk_count_for_size(size, self.file_chunk_size)):
                chunk = handle.read(self.file_chunk_size)
                key = model_file_chunk_key(checkpoint_id, relative_path, index)
                self._put(key, chunk, config)
                written.append(key)
                keys.append(key)
                total += len(chunk)
        if total != size:
            raise WeightStoreError(
                f"{relative_path}: read {total} bytes, expected {size}; "
                "the file changed during import"
            )
        return FileLayout(path=relative_path, size=size, chunked=True, keys=keys)

    def _store_attr(self, name: str):
        """Find a class the store's bindings expose, e.g. ReplicateConfig.

        Looked up on the store object first (so a test double can supply its
        own), then on the module the store's own class came from. Resolving it
        from the instance rather than a hard-coded import path keeps this
        working whichever way the extension was built or installed.
        """
        found = getattr(self.store, name, None)
        if found is not None:
            return found
        module = sys.modules.get(type(self.store).__module__)
        return getattr(module, name, None)

    def _config(self, *, chunked: bool):
        config_cls = self._store_attr("ReplicateConfig")
        if config_cls is None:
            raise WeightStoreError(
                "the store's bindings expose no ReplicateConfig; "
                "cannot describe how to place objects"
            )
        config = config_cls()
        config.replica_num = self.replica_num
        if hasattr(config, "with_hard_pin"):
            # Weights are few, large, long-lived and read many times, unlike
            # KVCache entries. Pinning them is how that difference in
            # lifetime policy is expressed. Note it prevents eviction, not
            # loss: a restarted node still drops its chunks.
            config.with_hard_pin = self.hard_pin_weights
        data_type_cls = self._store_attr("ObjectDataType")
        if data_type_cls is not None and hasattr(config, "data_type"):
            name = "WEIGHT" if chunked else "METADATA"
            if hasattr(data_type_cls, name):
                config.data_type = getattr(data_type_cls, name)
        return config

    def _put(self, key: str, value: bytes, config) -> None:
        result = self.store.put(key, value, config)
        if result not in (0, None):
            raise WeightStoreError(f"failed to put {key}: {result}")

    def _remove_by_regex(self, pattern: str) -> int:
        """Remove matching keys, ignoring read leases (see delete_model)."""
        try:
            removed = self.store.remove_by_regex(pattern, True)
        except TypeError:
            # A store (or test double) without the force parameter.
            removed = self.store.remove_by_regex(pattern)
        return 0 if removed is None else int(removed)

    def _get_optional(self, key: str) -> Optional[bytes]:
        value = self.store.get(key)
        if value is None:
            return None
        if isinstance(value, str):
            return value.encode("utf-8")
        return bytes(value)

    def _read_keys(self, keys: Iterable[str]) -> bytes:
        parts: List[bytes] = []
        for key in keys:
            value = self._get_optional(key)
            if value is None:
                raise IncompleteCheckpointError("", [key])
            parts.append(value)
        return b"".join(parts)

    def _missing_keys(self, keys: List[str]) -> List[str]:
        if not keys:
            return []
        batch = getattr(self.store, "batch_is_exist", None)
        if batch is None:
            return [key for key in keys if not self._exists(key)]
        results = batch(list(keys))
        if len(results) != len(keys):
            raise WeightStoreError(
                f"batch_is_exist returned {len(results)} results "
                f"for {len(keys)} keys"
            )
        missing: List[str] = []
        for key, result in zip(keys, results):
            if result < 0:
                raise WeightStoreError(f"failed to check {key}: {result}")
            if result == 0:
                missing.append(key)
        return missing

    def _exists(self, key: str) -> bool:
        checker = getattr(self.store, "is_exist", None)
        if checker is None:
            return self.store.get(key) is not None
        result = checker(key)
        if result < 0:
            raise WeightStoreError(f"failed to check {key}: {result}")
        return bool(result)

    def _remove_keys(self, keys: Iterable[str]) -> None:
        for key in keys:
            try:
                result = self.store.remove(key, True)
            except TypeError:
                result = self.store.remove(key)
            except BaseException:
                continue
            if result not in (0, None, MISSING_OBJECT_ERROR):
                self._log(f"warning: failed to remove {key}: {result}")

    def _log(self, message: str) -> None:
        if self.progress:
            print(message, file=sys.stderr, flush=True)


def _list_model_files(source_dir: Path) -> List[str]:
    """Relative paths of every regular file under a model directory, sorted."""
    paths: List[str] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source_dir).as_posix()
        try:
            validate_relative_path(relative)
        except ValueError:
            # A name that cannot be represented as a key (e.g. a "chunks"
            # directory) would silently alias a chunk key; refuse the import
            # rather than store something unreadable.
            raise WeightStoreError(
                f"cannot store {relative!r}: its path is not a valid key"
            ) from None
        paths.append(relative)
    return paths


def _format_bytes(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{value:.1f}TiB"
