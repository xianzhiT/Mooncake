"""Regression tests for the weight management contract (no inference engine)."""

import multiprocessing

import pytest
from mooncake.weight_store import (
    WeightCacheClient,
    WeightStoreError,
    model_file_chunk_key,
    weight_cache,
)
from test_weight_cache import WriteOnceStore, write_model


@pytest.fixture
def native_store():
    return WriteOnceStore()


@pytest.fixture
def client(native_store, tmp_path):
    return WeightCacheClient(
        native_store, progress=False, management_lock_dir=tmp_path / "locks"
    )


def test_native_missing_index_allows_single_shard(client, tmp_path):
    source = write_model(tmp_path / "source", single_file=True)
    assert client.import_model("single", source).complete


def test_native_missing_header_reports_incomplete(client, native_store, tmp_path):
    client.import_model("m", write_model(tmp_path / "source"))
    key = model_file_chunk_key("m", "model-00001-of-00002.safetensors", 0)
    del native_store.data[key]
    status = client.inspect_model("m")
    assert not status.complete
    assert key in status.missing_keys


def test_failed_get_of_existing_file_is_an_error(client, native_store, tmp_path):
    client.import_model("m", write_model(tmp_path / "source"))
    native_store.get = lambda key: b""
    with pytest.raises(WeightStoreError, match="read"):
        client.read_file("m", "tokenizer.json")


def test_missing_read_propagates_metadata_error(client, native_store):
    native_store.get_size = lambda key: -500
    with pytest.raises(WeightStoreError):
        client.read_file("m", "tokenizer.json")


def test_chunk_size_cannot_be_configured(native_store):
    with pytest.raises(TypeError):
        WeightCacheClient(native_store, file_chunk_size=1024)


def test_force_replacement_is_not_supported(client, native_store, tmp_path):
    source = write_model(tmp_path / "source")
    client.import_model("m", source)
    before = dict(native_store.data)
    with pytest.raises(TypeError):
        client.import_model("m", source, force=True)
    assert native_store.data == before


def test_failed_import_leaves_partial_data_for_explicit_cleanup(
    client, native_store, tmp_path
):
    put = native_store.put
    calls = 0

    def fail(key, value, config=None):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("write failed")
        return put(key, value, config)

    native_store.put = fail
    with pytest.raises(RuntimeError, match="write failed"):
        client.import_model("m", write_model(tmp_path / "source"))
    assert native_store.data
    assert client.delete_model("m") > 0
    assert not native_store.data


@pytest.mark.parametrize("name", ["pytorch_model.bin", "weights.pt", "model.gguf"])
def test_unsupported_weights_rejected_before_any_write(
    client, native_store, tmp_path, name
):
    source = write_model(tmp_path / "source", extra={name: b"weights"})
    with pytest.raises(WeightStoreError, match="safetensors"):
        client.import_model("m", source)
    assert not native_store.data


def test_source_missing_indexed_shard_is_rejected_before_write(
    client, native_store, tmp_path
):
    source = write_model(tmp_path / "source")
    (source / "model-00001-of-00002.safetensors").unlink()
    with pytest.raises(WeightStoreError):
        client.import_model("m", source)
    assert not native_store.data


def test_truncated_read_is_rejected(client, native_store, tmp_path):
    client.import_model("m", write_model(tmp_path / "source"))
    shard = "model-00001-of-00002.safetensors"
    key = model_file_chunk_key("m", shard, 0)
    native_store.data[key] = native_store.data[key][:-1]
    with pytest.raises(WeightStoreError):
        client.read_file("m", shard)


def test_auxiliary_files_are_not_part_of_weight_completeness(
    client, native_store, tmp_path
):
    client.import_model("m", write_model(tmp_path / "source"))
    del native_store.data["weight/models/m/files/tokenizer.json"]
    assert client.verify_model("m").complete


def _hold_import(lock_dir, source, connection):
    store = WriteOnceStore()

    def pause_query(pattern):
        connection.send("locked")
        connection.recv()
        raise RuntimeError("stop import")

    store.query_keys_by_regex = pause_query
    client = WeightCacheClient(store, progress=False, management_lock_dir=lock_dir)
    try:
        client.import_model("busy", source)
    except RuntimeError:
        pass
    finally:
        connection.close()


@pytest.mark.parametrize("terminate", [False, True])
def test_process_lock_blocks_both_mutations_and_releases_after_exit(
    tmp_path, terminate
):
    lock_dir = tmp_path / "locks"
    source = write_model(tmp_path / "source")
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_hold_import, args=(lock_dir, source, child))
    proc.start()
    child.close()
    try:
        assert parent.poll(10), "import never acquired its management lock"
        assert parent.recv() == "locked"
        client = WeightCacheClient(
            WriteOnceStore(), progress=False, management_lock_dir=lock_dir
        )
        for operation in (
            lambda: client.import_model("busy", source),
            lambda: client.delete_model("busy"),
        ):
            with pytest.raises(weight_cache.CheckpointBusyError):
                operation()
        # Another checkpoint is independent; readers do not acquire this lock.
        assert client.import_model("other", source).complete
        if terminate:
            proc.terminate()
        else:
            parent.send("finish")
        proc.join(10)
        assert not proc.is_alive()
        assert client.import_model("busy", source).complete
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(10)
        parent.close()


def test_query_failure_aborts_import_without_writing(client, native_store, tmp_path):
    def fail(pattern):
        raise RuntimeError("query failed")

    native_store.query_keys_by_regex = fail
    with pytest.raises(RuntimeError, match="query failed"):
        client.import_model("m", write_model(tmp_path / "source"))
    assert not native_store.data


def test_failed_materialization_preserves_destination(client, native_store, tmp_path):
    source = write_model(tmp_path / "source")
    client.import_model("m", source)
    shard = "model-00001-of-00002.safetensors"
    key = model_file_chunk_key("m", shard, 0)
    native_store.data[key] = native_store.data[key][:-1]
    destination = tmp_path / "out" / shard
    destination.parent.mkdir()
    destination.write_bytes(b"existing file")
    with pytest.raises(WeightStoreError):
        client.materialize_file("m", shard, destination)
    assert destination.read_bytes() == b"existing file"
    assert list(destination.parent.iterdir()) == [destination]


def test_delete_holds_same_lock_as_import(client, native_store, tmp_path):
    source = write_model(tmp_path / "source")
    client.import_model("m", source)
    remove = native_store.remove_by_regex

    def check_lock(pattern, force):
        with pytest.raises(weight_cache.CheckpointBusyError):
            client.import_model("m", source)
        # Reads remain available while deletion holds the management lock.
        assert client.inspect_model("m").complete
        return remove(pattern, force)

    native_store.remove_by_regex = check_lock
    assert client.delete_model("m") > 0


def test_max_length_checkpoint_can_be_managed(client, tmp_path):
    name = "a" * 255
    client.import_model(name, write_model(tmp_path / "source"))
    assert client.delete_model(name) > 0


def test_multi_chunk_header_and_missing_header_chunk(
    client, native_store, tmp_path, monkeypatch
):
    monkeypatch.setattr(weight_cache, "DEFAULT_FILE_CHUNK_SIZE", 1024)
    tensors = {f"tensor_{i}": 8 for i in range(40)}
    shard = "model-00001-of-00001.safetensors"
    source = write_model(tmp_path / "source", shards={shard: tensors})
    assert client.import_model("m", source).complete
    assert client.read_file("m", shard) == (source / shard).read_bytes()
    missing = model_file_chunk_key("m", shard, 1)
    del native_store.data[missing]
    status = client.inspect_model("m")
    assert not status.complete
    assert missing in status.missing_keys
