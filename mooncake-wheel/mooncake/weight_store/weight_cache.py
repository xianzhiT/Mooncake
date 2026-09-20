"""Safetensors weight file management on top of Mooncake Store.

Only file bytes live in Store. Layouts are derived from the original index
and shard headers. Completeness is an observation of weight availability,
not a serving-readiness guarantee or a checksum against the source.

All writers must run on one management host with the same lock directory.
Import and delete take a per-checkpoint OS file lock; readers remain unlocked.
Use a new checkpoint id for new content. Failed imports leave partial data
for explicit deletion, never roll back keys that another operation may use.
"""

from __future__ import annotations

import fcntl
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

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
DEFAULT_MANAGEMENT_LOCK_DIR = "/tmp/mooncake-weight-store-locks"
# Native get_size returns this for a missing object.
MISSING_OBJECT_ERROR = -704


class WeightStoreError(RuntimeError):
    """Base class for control-plane failures."""


class CheckpointBusyError(WeightStoreError):
    """Another management operation holds this checkpoint's local file lock."""


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

    def __init__(self, checkpoint_id: str, missing_keys: list[str]) -> None:
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
    keys: list[str]


@dataclass
class CheckpointStatus:
    """Observed weight availability, including config/index bootstrap files.

    Does not cover tokenizer/processor files, model startup, content hashes,
    or guarantee that objects remain available after inspection. If a header
    is missing, its shard size is unknown and total_size is a lower bound.
    """

    checkpoint_id: str
    files: list[FileLayout] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)

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
        hard_pin_weights: bool = True,
        progress: bool = True,
        management_lock_dir: str | os.PathLike[str] = DEFAULT_MANAGEMENT_LOCK_DIR,
    ) -> None:
        if replica_num <= 0:
            raise ValueError(f"invalid replica_num: {replica_num!r}")
        self.store = store
        self.replica_num = replica_num
        self.management_lock_dir = Path(management_lock_dir)
        self.hard_pin_weights = hard_pin_weights
        self.progress = progress

    # ---------------------------------------------------------------- import

    def import_model(
        self,
        checkpoint_id: str,
        source: str | os.PathLike[str],
    ) -> CheckpointStatus:
        """Mirror a model directory into the store.

        Files are copied as they are: weight shards chunked, everything else
        whole. No index or manifest is synthesised -- a single-file model that
        has no safetensors index is stored without one, exactly as on disk.
        """
        validate_checkpoint_id(checkpoint_id)
        with self._management_lock(checkpoint_id):
            source_dir = Path(source)
            relative_paths = self._validate_source(source_dir)
            existing = self.store.query_keys_by_regex(
                model_prefix_pattern(checkpoint_id)
            )
            if existing:
                raise CheckpointExistsError(
                    f"checkpoint {checkpoint_id!r} already has {len(existing)} object(s). "
                    "Use a new id for new content; explicitly delete partial imports "
                    "before retrying. Store put does not overwrite existing objects."
                )
            for index, relative_path in enumerate(relative_paths, start=1):
                layout = self._put_file(
                    checkpoint_id, source_dir / relative_path, relative_path
                )
                self._log(
                    f"[{index}/{len(relative_paths)}] {relative_path} "
                    f"({_format_bytes(layout.size)}, {len(layout.keys)} object(s))"
                )
            status = self.inspect_model(checkpoint_id)
            if not status.complete:
                raise IncompleteCheckpointError(checkpoint_id, status.missing_keys)
            return status

    # ------------------------------------------------------------- inspect

    def list_models(self) -> list[str]:
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
        expected: list[str] = []
        for layout in layouts:
            expected.extend(layout.keys)
        missing = self._missing_keys(expected)
        return CheckpointStatus(
            checkpoint_id=checkpoint_id,
            files=layouts,
            missing_keys=missing,
        )

    def verify_model(self, checkpoint_id: str) -> CheckpointStatus:
        """Read weight/bootstrap bytes back and check chunk lengths.

        Stronger than inspect_model, which only checks that keys exist. No
        stored checksum is compared, because none is stored; instead each
        safetensors header is parsed and its declared size checked against
        the bytes actually present, which catches truncation and corruption of
        the header region.
        """
        status = self.inspect_model(checkpoint_id)
        if not status.complete:
            raise IncompleteCheckpointError(checkpoint_id, status.missing_keys)

        for layout in status.files:
            # Stream one chunk at a time rather than duplicating a whole shard
            # in memory. inspect_model already parsed the shard's header.
            for _ in self._read_layout(checkpoint_id, layout):
                pass
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
        with self._management_lock(checkpoint_id):
            removed = self.store.remove_by_regex(
                model_prefix_pattern(checkpoint_id), True
            )
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
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                for part in self._read_layout(checkpoint_id, layout):
                    handle.write(part)
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return size

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
        return b"".join(self._read_layout(checkpoint_id, layout))

    # ------------------------------------------------------------ internals

    def _derive_layouts(self, checkpoint_id: str) -> list[FileLayout]:
        """Work out which files a checkpoint should have, and their keys.

        Reads the model's own metadata rather than a stored description:
        config.json proves the model is there, the safetensors index names the
        shards, and each shard's header gives its size.
        """
        config = self._get_optional(model_config_key(checkpoint_id))
        if config is None:
            raise CheckpointNotFoundError(
                f"checkpoint {checkpoint_id!r} is not in the store (no {CONFIG_FILE})"
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
            try:
                size = self._weight_shard_size(checkpoint_id, shard)
            except IncompleteCheckpointError as exc:
                layouts.append(
                    FileLayout(path=shard, size=0, chunked=True, keys=exc.missing_keys)
                )
                continue
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
            count = chunk_count_for_size(size, DEFAULT_FILE_CHUNK_SIZE)
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

    def _weight_shard_size(self, checkpoint_id: str, relative_path: str) -> int | None:
        """Size of a stored shard, read from its own header.

        Fetches the first chunk and, for large headers, subsequent chunks.
        The header declares the total size without reading the tensor payload.
        """
        _require_safetensors(relative_path)
        prefix = self._get_optional(
            model_file_chunk_key(checkpoint_id, relative_path, 0)
        )
        if prefix is None:
            return None
        header_end = 8 + safetensors_header.header_length(prefix)
        # A valid header may span more than one chunk. Its length, never a
        # missing key, determines where to stop reading.
        index = 1
        while len(prefix) < header_end:
            if len(prefix) != index * DEFAULT_FILE_CHUNK_SIZE:
                raise WeightStoreError(f"{relative_path}: truncated header chunk")
            key = model_file_chunk_key(checkpoint_id, relative_path, index)
            part = self._get_optional(key)
            if part is None:
                raise IncompleteCheckpointError(checkpoint_id, [key])
            prefix += part
            index += 1
        return safetensors_header.total_size(prefix)

    def _file_size(self, checkpoint_id: str, relative_path: str) -> int | None:
        if is_weight_file(relative_path):
            return self._weight_shard_size(checkpoint_id, relative_path)
        payload = self._get_optional(model_file_key(checkpoint_id, relative_path))
        return None if payload is None else len(payload)

    def _put_file(
        self,
        checkpoint_id: str,
        source_path: Path,
        relative_path: str,
    ) -> FileLayout:
        """Store one source file using the fixed chunk layout."""
        size = source_path.stat().st_size
        if not is_weight_file(relative_path):
            payload = source_path.read_bytes()
            key = model_file_key(checkpoint_id, relative_path)
            self._put(key, payload, self._config(chunked=False))
            return FileLayout(path=relative_path, size=size, chunked=False, keys=[key])

        config = self._config(chunked=True)
        keys: list[str] = []
        total = 0
        with source_path.open("rb") as handle:
            for index in range(chunk_count_for_size(size, DEFAULT_FILE_CHUNK_SIZE)):
                chunk = handle.read(DEFAULT_FILE_CHUNK_SIZE)
                key = model_file_chunk_key(checkpoint_id, relative_path, index)
                self._put(key, chunk, config)
                keys.append(key)
                total += len(chunk)
        if total != size or source_path.stat().st_size != size:
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

    def _get_optional(self, key: str) -> bytes | None:
        value = self.store.get(key)
        if value:
            return bytes(value)
        # Native get returns b"" for both failure and an empty object. Only a
        # missing-object result from metadata permits treating it as absent.
        size = self.store.get_size(key)
        if size == MISSING_OBJECT_ERROR:
            return None
        if size == 0 and value is not None:
            return b""
        raise WeightStoreError(f"failed to read {key}: get_size returned {size}")

    def _read_layout(self, checkpoint_id: str, layout: FileLayout) -> Iterator[bytes]:
        remaining = layout.size
        for key in layout.keys:
            value = self._get_optional(key)
            if value is None:
                raise IncompleteCheckpointError(checkpoint_id, [key])
            expected = (
                min(remaining, DEFAULT_FILE_CHUNK_SIZE) if layout.chunked else remaining
            )
            if len(value) != expected:
                raise WeightStoreError(
                    f"{layout.path}: expected {expected} bytes in {key}, read {len(value)}"
                )
            remaining -= len(value)
            yield value

    def _missing_keys(self, keys: list[str]) -> list[str]:
        if not keys:
            return []
        batch = getattr(self.store, "batch_is_exist", None)
        if batch is None:
            return [key for key in keys if not self._exists(key)]
        results = batch(list(keys))
        if len(results) != len(keys):
            raise WeightStoreError(
                f"batch_is_exist returned {len(results)} results for {len(keys)} keys"
            )
        missing: list[str] = []
        for key, result in zip(keys, results):
            if result < 0:
                raise WeightStoreError(f"failed to check {key}: {result}")
            if result == 0:
                missing.append(key)
        return missing

    def _exists(self, key: str) -> bool:
        checker = getattr(self.store, "is_exist", None)
        if checker is None:
            return self._get_optional(key) is not None
        result = checker(key)
        if result < 0:
            raise WeightStoreError(f"failed to check {key}: {result}")
        return bool(result)

    @contextmanager
    def _management_lock(self, checkpoint_id: str):
        """Cooperative, nonblocking lock on one management host.

        Never unlink lock files: waiters and new callers must lock the same
        inode. All managers must share this directory and filesystem namespace.
        This does not coordinate writers on different hosts or raw Store puts.
        """
        self.management_lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        with (self.management_lock_dir / checkpoint_id).open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CheckpointBusyError(
                    f"checkpoint {checkpoint_id!r} has an active management operation"
                ) from exc
            # Closing the descriptor releases the lock, including on exceptions.
            yield

    def _validate_source(self, source_dir: Path) -> list[str]:
        if not source_dir.is_dir():
            raise WeightStoreError(f"source is not a directory: {source_dir}")
        paths = _list_model_files(source_dir)
        if CONFIG_FILE not in paths:
            raise WeightStoreError(f"{source_dir} has no {CONFIG_FILE}")
        shards = {path for path in paths if is_weight_file(path)}
        for path in shards:
            _require_safetensors(path)
        if SAFETENSORS_INDEX_FILE in paths:
            expected = set(
                safetensors_header.weight_map_files(
                    (source_dir / SAFETENSORS_INDEX_FILE).read_bytes()
                )
            )
        else:
            expected = {"model.safetensors"}
        if shards != expected:
            raise WeightStoreError(
                "safetensors files must match the index, or consist of model.safetensors "
                f"without an index; missing={sorted(expected - shards)}, "
                f"unlisted={sorted(shards - expected)}"
            )
        for shard in sorted(shards):
            path = source_dir / shard
            with path.open("rb") as handle:
                prefix = handle.read(8)
                length = safetensors_header.header_length(prefix)
                declared = safetensors_header.total_size(prefix + handle.read(length))
            if declared != path.stat().st_size:
                raise WeightStoreError(
                    f"{shard}: source size does not match its header"
                )
        return paths

    def _log(self, message: str) -> None:
        if self.progress:
            print(message, file=sys.stderr, flush=True)


def _list_model_files(source_dir: Path) -> list[str]:
    """Relative paths of every regular file under a model directory, sorted."""
    paths: list[str] = []
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


def _require_safetensors(path: str) -> None:
    validate_relative_path(path)
    if not path.endswith(".safetensors"):
        raise WeightStoreError(
            f"unsupported weight format: {path}; only safetensors is supported"
        )
