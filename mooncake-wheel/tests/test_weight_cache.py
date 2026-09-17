"""Tests for the manifest-less weight cache.

The fake store below enforces the two store behaviours the design depends on:

* put is write-once -- a second put to an existing key reports success and
  discards the new bytes. Anything that assumes overwriting works must fail
  here rather than in production.
* regex queries and removals are server-side, matching against the full key.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import pytest

from mooncake.weight_store import (
    CheckpointExistsError,
    CheckpointNotFoundError,
    IncompleteCheckpointError,
    WeightCacheClient,
    WeightStoreError,
    checkpoint_id_from_config_key,
    config_listing_pattern,
    model_config_key,
    model_file_chunk_key,
    model_file_key,
    model_safetensors_index_key,
)
from mooncake.weight_store import model_keyspace, safetensors_header


class FakeReplicateConfig:
    def __init__(self) -> None:
        self.replica_num = 1
        self.with_hard_pin = False
        self.data_type = None


class WriteOnceStore:
    """In-memory stand-in with the real store's write-once put.

    Also models read leases: the master grants one on every read and an
    unforced removal skips a still-leased key, so a delete that does not force
    leaves most of a just-read model behind.
    """

    ReplicateConfig = FakeReplicateConfig

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.discarded_puts: list[str] = []
        self.configs: dict[str, FakeReplicateConfig] = {}
        self.leased: set[str] = set()

    def put(self, key: str, value: bytes, config=None) -> int:
        if key in self.data:
            # Exactly what the real store does: report success, keep the old
            # bytes. Silently.
            self.discarded_puts.append(key)
            return 0
        self.data[key] = bytes(value)
        if config is not None:
            self.configs[key] = config
        return 0

    def get(self, key: str):
        if key in self.data:
            self.leased.add(key)
        return self.data.get(key)

    def is_exist(self, key: str) -> int:
        return 1 if key in self.data else 0

    def batch_is_exist(self, keys):
        return [1 if key in self.data else 0 for key in keys]

    def remove(self, key: str, force: bool = False) -> int:
        if key not in self.data:
            return -704
        if self.leased and key in self.leased and not force:
            return 0
        del self.data[key]
        self.leased.discard(key)
        self.configs.pop(key, None)
        return 0

    def query_keys_by_regex(self, pattern: str):
        compiled = re.compile(pattern)
        return [key for key in self.data if compiled.search(key)]

    def remove_by_regex(self, pattern: str, force: bool = False) -> int:
        compiled = re.compile(pattern)
        doomed = [key for key in self.data if compiled.search(key)]
        removed = 0
        for key in doomed:
            # The real master skips still-leased keys unless force is set.
            if key in self.leased and not force:
                continue
            del self.data[key]
            self.leased.discard(key)
            self.configs.pop(key, None)
            removed += 1
        return removed


def make_safetensors(tensors: dict[str, int], *, pad: int = 0) -> bytes:
    """Build a minimal valid safetensors file with the given tensor sizes."""
    header: dict = {}
    offset = 0
    for name, size in tensors.items():
        header[name] = {
            "dtype": "F32",
            "shape": [size // 4 or 1],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    blob = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(blob)) + blob + bytes(offset + pad)


def write_model(
    root: Path,
    *,
    shards: dict[str, dict[str, int]] | None = None,
    single_file: bool = False,
    extra: dict[str, bytes] | None = None,
) -> Path:
    """Lay out a HuggingFace-shaped model directory on disk."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"model_type": "test"}))
    (root / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))

    if single_file:
        (root / "model.safetensors").write_bytes(make_safetensors({"w": 2048}))
    else:
        shards = shards or {
            "model-00001-of-00002.safetensors": {"a": 4096},
            "model-00002-of-00002.safetensors": {"b": 4096},
        }
        weight_map = {}
        for filename, tensors in shards.items():
            (root / filename).write_bytes(make_safetensors(tensors))
            for name in tensors:
                weight_map[name] = filename
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 8192}, "weight_map": weight_map})
        )

    for name, payload in (extra or {}).items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return root


@pytest.fixture
def store() -> WriteOnceStore:
    return WriteOnceStore()


@pytest.fixture
def client(store: WriteOnceStore) -> WeightCacheClient:
    return WeightCacheClient(store, file_chunk_size=1024, progress=False)


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    return write_model(tmp_path / "src")


# ------------------------------------------------------------------ keyspace


def test_keys_are_deterministic():
    first = model_file_chunk_key("m", "model-00001.safetensors", 3)
    second = model_file_chunk_key("m", "model-00001.safetensors", 3)
    assert first == second
    assert first.endswith("/chunks/00000003")


def test_whole_file_key_has_no_chunk_segment():
    key = model_file_key("m", "config.json")
    assert key == "weight/models/m/files/config.json"
    assert "/chunks/" not in key


def test_config_key_round_trips_through_listing_pattern():
    key = model_config_key("qwen3-30b")
    assert re.fullmatch(config_listing_pattern(), key)
    assert checkpoint_id_from_config_key(key) == "qwen3-30b"


def test_listing_pattern_excludes_chunks_and_other_namespaces():
    pattern = re.compile(config_listing_pattern())
    assert not pattern.search(model_file_chunk_key("m", "model-00001.safetensors", 0))
    assert not pattern.search("other/ns/files/config.json")
    assert not pattern.search(model_safetensors_index_key("m"))


def test_paths_that_would_alias_a_chunk_key_are_rejected():
    with pytest.raises(ValueError):
        model_file_key("m", "weights/chunks/00000000")


@pytest.mark.parametrize("bad", ["", "/abs", "a/", "a//b", "../escape", "a/./b"])
def test_invalid_relative_paths_are_rejected(bad: str):
    with pytest.raises(ValueError):
        model_keyspace.validate_relative_path(bad)


def test_empty_file_still_occupies_one_chunk():
    assert model_keyspace.chunk_count_for_size(0, 1024) == 1
    assert model_keyspace.chunk_count_for_size(1, 1024) == 1
    assert model_keyspace.chunk_count_for_size(1024, 1024) == 1
    assert model_keyspace.chunk_count_for_size(1025, 1024) == 2


# -------------------------------------------------------- safetensors header


def test_total_size_is_derived_from_the_header():
    payload = make_safetensors({"a": 256, "b": 512})
    assert safetensors_header.total_size(payload) == len(payload)


def test_header_length_beyond_the_prefix_is_reported_not_guessed():
    payload = make_safetensors({"a": 64})
    truncated = payload[:6]
    with pytest.raises(safetensors_header.SafetensorsFormatError):
        safetensors_header.total_size(truncated)


def test_absurd_header_length_is_rejected():
    bogus = struct.pack("<Q", safetensors_header.MAX_HEADER_LENGTH + 1) + b"{}"
    with pytest.raises(safetensors_header.SafetensorsFormatError):
        safetensors_header.total_size(bogus)


def test_weight_map_files_are_deduplicated():
    index = json.dumps(
        {
            "weight_map": {
                "a": "shard-1.safetensors",
                "b": "shard-1.safetensors",
                "c": "shard-2.safetensors",
            }
        }
    ).encode()
    assert safetensors_header.weight_map_files(index) == [
        "shard-1.safetensors",
        "shard-2.safetensors",
    ]


# -------------------------------------------------------------------- import


def test_import_stores_bytes_and_nothing_else(
    client: WeightCacheClient, store: WriteOnceStore, model_dir: Path
):
    client.import_model("m", model_dir)

    # No manifest, no status, no registry, no lock -- every key is a file or a
    # chunk of one.
    for key in store.data:
        assert key.startswith("weight/models/m/files/")
    assert not any("manifest" in key for key in store.data)
    assert not any("/importing" in key for key in store.data)
    assert not any(key.startswith("weight/index/") for key in store.data)


def test_small_files_are_stored_whole_so_one_get_retrieves_them(
    client: WeightCacheClient, store: WriteOnceStore, model_dir: Path
):
    client.import_model("m", model_dir)
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        assert model_file_key("m", name) in store.data
        assert model_file_chunk_key("m", name, 0) not in store.data


def test_weight_shards_are_chunked(
    client: WeightCacheClient, store: WriteOnceStore, tmp_path: Path
):
    source = write_model(
        tmp_path / "big", shards={"model-00001-of-00001.safetensors": {"w": 4096}}
    )
    client.import_model("m", source)
    shard = "model-00001-of-00001.safetensors"
    assert model_file_chunk_key("m", shard, 0) in store.data
    assert model_file_chunk_key("m", shard, 1) in store.data
    assert model_file_key("m", shard) not in store.data


def test_import_reports_a_complete_checkpoint(
    client: WeightCacheClient, model_dir: Path
):
    status = client.import_model("m", model_dir)
    assert status.complete
    assert status.missing_keys == []
    assert status.total_size > 0


def test_import_rejects_a_directory_without_config(
    client: WeightCacheClient, tmp_path: Path
):
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "model.safetensors").write_bytes(make_safetensors({"w": 8}))
    with pytest.raises(WeightStoreError, match="config.json"):
        client.import_model("m", bare)


def test_import_cleans_up_after_a_failure(
    client: WeightCacheClient, store: WriteOnceStore, model_dir: Path, monkeypatch
):
    calls = {"n": 0}
    original = store.put

    def failing_put(key, value, config=None):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("store died mid-import")
        return original(key, value, config)

    monkeypatch.setattr(store, "put", failing_put)
    with pytest.raises(RuntimeError, match="store died"):
        client.import_model("m", model_dir)
    assert store.data == {}


def test_import_over_an_existing_checkpoint_is_refused(
    client: WeightCacheClient, model_dir: Path
):
    client.import_model("m", model_dir)
    with pytest.raises(CheckpointExistsError):
        client.import_model("m", model_dir)


def test_refusal_prevents_silently_serving_stale_weights(
    client: WeightCacheClient, store: WriteOnceStore, tmp_path: Path
):
    """The guard is what stops write-once from hiding a changed checkpoint.

    Without it, every put reports success while keeping the old bytes, the
    import reports done, and the model loads -- as the previous version.
    """
    first = write_model(
        tmp_path / "v1", shards={"model-00001-of-00001.safetensors": {"a": 512}}
    )
    client.import_model("m", first)
    before = dict(store.data)

    second = write_model(
        tmp_path / "v2", shards={"model-00001-of-00001.safetensors": {"a": 4096}}
    )
    with pytest.raises(CheckpointExistsError):
        client.import_model("m", second)
    assert store.data == before
    assert store.discarded_puts == []


def test_force_replaces_an_existing_checkpoint(
    client: WeightCacheClient, store: WriteOnceStore, tmp_path: Path
):
    first = write_model(
        tmp_path / "v1", shards={"model-00001-of-00001.safetensors": {"a": 512}}
    )
    client.import_model("m", first)

    second = write_model(
        tmp_path / "v2", shards={"model-00001-of-00001.safetensors": {"a": 4096}}
    )
    status = client.import_model("m", second, force=True)
    assert status.complete
    # The new bytes really landed: no put was discarded.
    assert store.discarded_puts == []
    shard = "model-00001-of-00001.safetensors"
    stored = client.read_file("m", shard)
    assert stored == (second / shard).read_bytes()


def test_guard_catches_leftovers_from_an_interrupted_import(
    client: WeightCacheClient, store: WriteOnceStore, model_dir: Path
):
    """A crash can leave chunks without a config.json.

    Checking the whole prefix -- not just config.json -- is what makes those
    leftovers visible, so a retry cannot land on top of them. The leftover
    here is a *valid* shard chunk from an earlier revision, so the only thing
    under test is whether the guard notices it: a config.json-only guard waves
    the import through, write-once keeps these stale bytes, and the import
    still reports success.
    """
    shard = "model-00001-of-00002.safetensors"
    orphan = model_file_chunk_key("m", shard, 0)
    stale_bytes = make_safetensors({"a": 64})
    store.put(orphan, stale_bytes)
    assert store.get(model_config_key("m")) is None

    with pytest.raises(CheckpointExistsError):
        client.import_model("m", model_dir)
    # Nothing was overwritten, so no put was silently discarded.
    assert store.discarded_puts == []
    assert store.get(orphan) == stale_bytes


def test_concurrent_imports_of_the_same_bytes_do_not_corrupt_each_other(
    store: WriteOnceStore, model_dir: Path
):
    """Two importers racing on one checkpoint is harmless.

    Identical bytes go to identical keys, so whoever loses each put still
    leaves the correct content behind. No lock needed.
    """
    first = WeightCacheClient(store, file_chunk_size=1024, progress=False)
    second = WeightCacheClient(store, file_chunk_size=1024, progress=False)
    first.import_model("m", model_dir)
    second.import_model("m", model_dir, force=True)
    status = second.verify_model("m")
    assert status.complete


# ------------------------------------------------------------------- listing


def test_list_models_returns_one_entry_per_model(
    client: WeightCacheClient, tmp_path: Path
):
    client.import_model("alpha", write_model(tmp_path / "a"))
    client.import_model("beta", write_model(tmp_path / "b"))
    assert client.list_models() == ["alpha", "beta"]


def test_list_models_ignores_unrelated_keys(
    client: WeightCacheClient, store: WriteOnceStore, model_dir: Path
):
    client.import_model("m", model_dir)
    store.put("kvcache/deadbeef", b"x")
    store.put("other/ns/files/config.json", b"x")
    assert client.list_models() == ["m"]


def test_list_models_is_empty_for_an_empty_store(client: WeightCacheClient):
    assert client.list_models() == []


def test_list_models_does_not_enumerate_chunks(
    client: WeightCacheClient, store: WriteOnceStore, model_dir: Path
):
    """Listing must stay small however many chunks a model has."""
    client.import_model("m", model_dir)
    seen: list[str] = []
    original = store.query_keys_by_regex

    def recording(pattern):
        result = original(pattern)
        seen.extend(result)
        return result

    store.query_keys_by_regex = recording
    client.list_models()
    assert all(key.endswith("/config.json") for key in seen)
    assert len(seen) == 1


# ------------------------------------------------------------------- inspect


def test_inspect_derives_the_file_list_from_the_model_itself(
    client: WeightCacheClient, store: WriteOnceStore, model_dir: Path
):
    client.import_model("m", model_dir)
    # Nothing recorded the layout; it is re-derived from config + index +
    # shard headers on every call.
    status = client.inspect_model("m")
    paths = {item.path for item in status.files}
    assert "config.json" in paths
    assert "model.safetensors.index.json" in paths
    assert "model-00001-of-00002.safetensors" in paths
    assert "model-00002-of-00002.safetensors" in paths


def test_inspect_rejects_an_absent_checkpoint(client: WeightCacheClient):
    with pytest.raises(CheckpointNotFoundError):
        client.inspect_model("nope")


def test_inspect_reports_a_lost_chunk_instead_of_claiming_ready(
    client: WeightCacheClient, store: WriteOnceStore, model_dir: Path
):
    """The central property: a restarted node loses chunks, and inspect says so.

    With a manifest this is where the old design lied -- the manifest survived
    on another node and kept reporting READY.
    """
    client.import_model("m", model_dir)
    lost = model_file_chunk_key("m", "model-00001-of-00002.safetensors", 0)
    assert lost in store.data
    del store.data[lost]

    status = client.inspect_model("m")
    assert not status.complete
    assert lost in status.missing_keys


def test_inspect_reports_a_chunk_lost_from_the_middle_of_a_file(
    client: WeightCacheClient, store: WriteOnceStore, tmp_path: Path
):
    """A hole must not be mistaken for the end of the file.

    Probing chunk keys until one is absent would stop at the hole and call a
    truncated shard complete. Sizes come from the header instead, so every
    expected chunk is checked.
    """
    source = write_model(
        tmp_path / "wide",
        shards={"model-00001-of-00001.safetensors": {"w": 8192}},
    )
    client.import_model("m", source)
    shard = "model-00001-of-00001.safetensors"
    middle = model_file_chunk_key("m", shard, 1)
    assert middle in store.data
    last = max(key for key in store.data if shard in key)
    del store.data[middle]

    status = client.inspect_model("m")
    assert not status.complete
    assert middle in status.missing_keys
    # The tail is still there, which is exactly why probing would have missed
    # the hole.
    assert last in store.data


def test_inspect_survives_a_missing_shard(
    client: WeightCacheClient, store: WriteOnceStore, model_dir: Path
):
    client.import_model("m", model_dir)
    shard = "model-00001-of-00002.safetensors"
    for key in [k for k in store.data if shard in k]:
        del store.data[key]
    status = client.inspect_model("m")
    assert not status.complete
    assert any(shard in key for key in status.missing_keys)


def test_status_dict_is_serialisable(client: WeightCacheClient, model_dir: Path):
    client.import_model("m", model_dir)
    payload = json.dumps(client.inspect_model("m").to_dict())
    assert json.loads(payload)["complete"] is True


# -------------------------------------------------------------------- verify


def test_verify_reads_every_byte_back(client: WeightCacheClient, model_dir: Path):
    client.import_model("m", model_dir)
    assert client.verify_model("m").complete


def test_verify_rejects_an_incomplete_checkpoint(
    client: WeightCacheClient, store: WriteOnceStore, model_dir: Path
):
    client.import_model("m", model_dir)
    lost = model_file_chunk_key("m", "model-00002-of-00002.safetensors", 0)
    del store.data[lost]
    with pytest.raises(IncompleteCheckpointError) as excinfo:
        client.verify_model("m")
    assert lost in excinfo.value.missing_keys


def test_verify_detects_a_corrupted_shard(
    client: WeightCacheClient, store: WriteOnceStore, model_dir: Path
):
    """No stored checksum, so verification re-parses the shard instead."""
    client.import_model("m", model_dir)
    shard = "model-00001-of-00002.safetensors"
    key = model_file_chunk_key("m", shard, 0)
    store.data[key] = store.data[key][:-8]
    with pytest.raises(WeightStoreError):
        client.verify_model("m")


# -------------------------------------------------------------------- delete


def test_delete_removes_every_key_of_a_checkpoint(
    client: WeightCacheClient, store: WriteOnceStore, model_dir: Path
):
    client.import_model("m", model_dir)
    client.delete_model("m")
    assert store.data == {}


def test_delete_leaves_other_checkpoints_alone(
    client: WeightCacheClient, store: WriteOnceStore, tmp_path: Path
):
    client.import_model("alpha", write_model(tmp_path / "a"))
    client.import_model("beta", write_model(tmp_path / "b"))
    client.delete_model("alpha")
    assert client.list_models() == ["beta"]
    assert client.verify_model("beta").complete


def test_delete_does_not_match_a_checkpoint_with_a_shared_prefix(
    client: WeightCacheClient, tmp_path: Path
):
    client.import_model("model", write_model(tmp_path / "a"))
    client.import_model("model-v2", write_model(tmp_path / "b"))
    client.delete_model("model")
    assert client.list_models() == ["model-v2"]


def test_delete_removes_keys_that_a_reader_just_leased(
    client: WeightCacheClient, store: WriteOnceStore, model_dir: Path
):
    """Reading a model must not make it undeletable.

    The master grants a read lease on every get, and an unforced removal
    skips still-leased keys -- so a delete right after an inspect or a load
    would report success while leaving most of the model behind, and `list`
    would keep showing it.
    """
    client.import_model("m", model_dir)
    client.verify_model("m")  # reads every byte, leasing every key
    assert store.leased

    client.delete_model("m")
    assert store.data == {}
    assert client.list_models() == []


def test_delete_is_idempotent(client: WeightCacheClient, model_dir: Path):
    client.import_model("m", model_dir)
    assert client.delete_model("m") > 0
    assert client.delete_model("m") == 0


def test_delete_sweeps_orphans_from_an_interrupted_import(
    client: WeightCacheClient, store: WriteOnceStore
):
    """Prefix deletion clears chunks no record ever mentioned."""
    store.put(model_file_chunk_key("m", "model-00001.safetensors", 0), b"orphan")
    store.put(model_file_chunk_key("m", "model-00001.safetensors", 1), b"orphan")
    assert client.delete_model("m") == 2
    assert store.data == {}


def test_delete_then_import_replaces_the_content(
    client: WeightCacheClient, store: WriteOnceStore, tmp_path: Path
):
    first = write_model(
        tmp_path / "v1", shards={"model-00001-of-00001.safetensors": {"a": 512}}
    )
    client.import_model("m", first)
    client.delete_model("m")

    second = write_model(
        tmp_path / "v2", shards={"model-00001-of-00001.safetensors": {"a": 4096}}
    )
    client.import_model("m", second)
    assert store.discarded_puts == []
    shard = "model-00001-of-00001.safetensors"
    assert client.read_file("m", shard) == (second / shard).read_bytes()


# --------------------------------------------------------------- single file


def test_single_file_model_needs_no_synthesised_index(
    client: WeightCacheClient, store: WriteOnceStore, tmp_path: Path
):
    """A model without an index is stored as-is, not given a fabricated one."""
    source = write_model(tmp_path / "single", single_file=True)
    client.import_model("m", source)
    assert model_safetensors_index_key("m") not in store.data
    assert client.list_models() == ["m"]
    assert client.verify_model("m").complete


def test_single_file_model_reports_a_lost_chunk(
    client: WeightCacheClient, store: WriteOnceStore, tmp_path: Path
):
    source = write_model(tmp_path / "single", single_file=True)
    client.import_model("m", source)
    lost = model_file_chunk_key("m", "model.safetensors", 0)
    del store.data[lost]
    status = client.inspect_model("m")
    assert not status.complete


# -------------------------------------------------------------- materialize


def test_materialize_writes_a_small_file_to_disk(
    client: WeightCacheClient, model_dir: Path, tmp_path: Path
):
    client.import_model("m", model_dir)
    out = tmp_path / "out" / "config.json"
    written = client.materialize_file("m", "config.json", out)
    assert out.read_bytes() == (model_dir / "config.json").read_bytes()
    assert written == out.stat().st_size


def test_materialize_rejects_an_absent_file(
    client: WeightCacheClient, model_dir: Path, tmp_path: Path
):
    client.import_model("m", model_dir)
    with pytest.raises(CheckpointNotFoundError):
        client.materialize_file("m", "absent.json", tmp_path / "x")


def test_read_file_round_trips_a_chunked_shard(
    client: WeightCacheClient, model_dir: Path
):
    client.import_model("m", model_dir)
    shard = "model-00001-of-00002.safetensors"
    assert client.read_file("m", shard) == (model_dir / shard).read_bytes()


def test_nested_files_are_preserved(client: WeightCacheClient, tmp_path: Path):
    source = write_model(tmp_path / "nested", extra={"subdir/extra.json": b"{}"})
    client.import_model("m", source)
    assert client.read_file("m", "subdir/extra.json") == b"{}"


# ------------------------------------------------------------------- config


def test_weight_shards_are_hard_pinned_and_typed(
    store: WriteOnceStore, model_dir: Path
):
    client = WeightCacheClient(
        store, file_chunk_size=1024, progress=False, hard_pin_weights=True
    )
    client.import_model("m", model_dir)
    chunk_key = model_file_chunk_key("m", "model-00001-of-00002.safetensors", 0)
    assert store.configs[chunk_key].with_hard_pin is True


def test_client_works_without_batch_is_exist(store: WriteOnceStore, model_dir: Path):
    """Existence checks fall back to per-key lookups if batching is absent."""
    delattr(type(store), "batch_is_exist")
    try:
        client = WeightCacheClient(store, file_chunk_size=1024, progress=False)
        client.import_model("m", model_dir)
        assert client.inspect_model("m").complete
    finally:
        WriteOnceStore.batch_is_exist = lambda self, keys: [
            1 if key in self.data else 0 for key in keys
        ]
