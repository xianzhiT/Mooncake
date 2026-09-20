# Weight file management

`WeightCacheClient` stores a local HuggingFace safetensors checkpoint in
Mooncake Store. Store holds only file bytes; the Python utility derives the
weight layout from the original index and safetensors headers. It does not
manage inference-engine readiness.

## Management contract

- Run all imports and deletes on **one management host**, under the same
  account and filesystem namespace, using the same `management_lock_dir`.
  The default is `/tmp/mooncake-weight-store-locks`. For long-running
  deployments, use a dedicated directory outside automatic temporary-file
  cleanup. Containers must share its mount.
- Import and delete acquire a nonblocking OS file lock per checkpoint. A
  conflicting operation raises `CheckpointBusyError`. Other checkpoint IDs
  remain independent. Locks are released when the operation exits or its
  process dies; do not unlink the lock files, including while idle.
- This is cooperative local coordination, **not a distributed lock**. Raw
  Store writes, another host, or another lock directory bypass it. Multiple
  remote callers must route mutations through the one management host.
- A checkpoint ID identifies immutable content. Use a new ID for a new
  version. Keep the source directory unchanged throughout import. The tool
  rejects any existing keys under the checkpoint prefix; it has no `force`
  replacement option.
- Failed imports leave partial data. Stop readers, explicitly delete the
  partial checkpoint, then retry the same source. There is no automatic
  rollback or persistent import-status record.
- Readers do not take management locks. Explicit deletion ignores Store read
  leases and can interrupt a concurrent reader. Inspection is an observation,
  not a reservation or an atomic snapshot of a whole checkpoint.

## File layout and supported inputs

The current layout fixes weight chunks at **64 MiB**. It is not configurable
per client. Earlier experimental imports using other chunk sizes must be
removed and re-imported before use.

```text
weight/models/<checkpoint>/files/config.json
weight/models/<checkpoint>/files/model.safetensors.index.json
weight/models/<checkpoint>/files/<shard>.safetensors/chunks/00000000
weight/models/<checkpoint>/files/<shard>.safetensors/chunks/00000001
```

A source must contain `config.json` and either:

- `model.safetensors` without an index; or
- `model.safetensors.index.json`, whose `weight_map` names exactly the
  safetensors files in the source directory.

The importer checks shard headers and declared sizes before writing. Other
weight formats (`.bin`, `.pt`, `.pth`, `.gguf`) are rejected. Auxiliary files
are copied whole and can be fetched by path. No synthetic index, manifest,
model registry, or lock object is written to Store.

## Operations

```python
from mooncake.weight_store import WeightCacheClient

# store is an already configured MooncakeDistributedStore.
client = WeightCacheClient(
    store,
    management_lock_dir="/var/lib/mooncake/weight-locks",
)
client.import_model("qwen-revision-a", "/models/qwen-revision-a")
print(client.list_models())
print(client.inspect_model("qwen-revision-a").to_dict())
client.verify_model("qwen-revision-a")
client.materialize_file("qwen-revision-a", "tokenizer.json", "/tmp/tokenizer.json")
client.delete_model("qwen-revision-a")
```

- `list_models`: lists checkpoints whose `config.json` exists. It is not a
  list of complete checkpoints and may omit partial imports with no config.
  Listing and import preflight scan Store metadata by regex. Query failures
  raise errors, never masquerade as an empty result.
- `inspect_model`: checks the expected weight keys and the config/index files
  needed to locate them. `complete` does **not** cover tokenizer, processor,
  custom code, or serving readiness. A missing config raises
  `CheckpointNotFoundError`; an unavailable index cannot reconstruct the
  original shard list. Missing shard headers make sizes unknown, so
  `total_size` is then a lower bound and `missing_keys` is not an exhaustive
  inventory of all lost chunks.
- `verify_model`: reads the inspected files one chunk at a time and checks
  exact lengths. It detects missing or truncated data; it does not compare
  against source hashes or prove tensor contents unchanged.
- `read_file` / `materialize_file`: reject missing chunks and incorrect chunk
  lengths. Materialization replaces the destination only after a complete
  read. `read_file` intentionally returns a whole file in memory.
- `delete_model`: removes the entire checkpoint prefix, including orphan
  chunks. Repeating deletion is safe once no importer is active.

Inspect currently fetches full chunks covering each shard header (normally
one chunk per shard). It is a management operation, not a cheap serving
health probe. No loading-performance claim is made by this module.

## Verification

Run `test_weight_cache.py` and `test_weight_cache_contract.py` with pytest.
The latter includes independent-process locking and process-death tests.
`test_weight_cache_native.py` provides opt-in tests against a dedicated TCP
master and a freshly built `mooncake.store` extension:

```bash
MOONCAKE_WEIGHT_TEST_MASTER=127.0.0.1:50051 \
    python -m unittest discover -s mooncake-wheel/tests -p test_weight_cache_native.py
```

Native checks use unique checkpoint IDs and remove their own objects. They
exercise the production 64 MiB layout without requiring SGLang or a GPU.

Set `MOONCAKE_WEIGHT_TEST_REAL_CLIENT=host:port` as well to run the native
suite through a separately started `mooncake_client` service (DummyClient
forwarding), using the same dedicated master.
