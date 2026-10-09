#!/usr/bin/env python3
"""P1 end-to-end: import a synthetic model dir -> open -> verify -> delete.

Validates, against a real master, the parts the authority only checks at commit:
  - payload_keys_sha256 (digest algo)
  - manifest_key (path encoding)
  - the 12 commit invariants (data_type, counts, affinity, logical_bytes, membership)
Then zero-copy reads every file back and checks SHA-256, and deletes the revision.

Run inside mooncake-fw:dev with a master running. file_weight is loaded as a
standalone top-level package to avoid the installed mooncake package __init__.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile

# This script lives inside file_weight/, whose sibling store.py would shadow the
# C++ `store` module. Drop the script's own dir from sys.path BEFORE importing it.
_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _SELF_DIR]

import store  # type: ignore  # the C++ pybind module

# Load file_weight as a standalone top-level package (bypass mooncake/__init__).
_SRC = "/src/mooncake-wheel/mooncake/file_weight"
_PKG_PARENT = tempfile.mkdtemp()
shutil.copytree(_SRC, os.path.join(_PKG_PARENT, "file_weight"))
sys.path.insert(0, _PKG_PARENT)
import file_weight  # noqa: E402
from file_weight import WeightRevisionIdentity  # noqa: E402
from file_weight.store import FileWeightStore  # noqa: E402

MASTER = os.environ.get("MOONCAKE_MASTER", "127.0.0.1:50181")


def _make_model_dir() -> str:
    d = tempfile.mkdtemp(prefix="model-")
    # a weight shard (>chunk to exercise multi-chunk), a small shard, and aux files
    with open(os.path.join(d, "model-00001-of-00002.safetensors"), "wb") as f:
        f.write(os.urandom(100))
    with open(os.path.join(d, "model-00002-of-00002.safetensors"), "wb") as f:
        f.write(os.urandom(50))
    with open(os.path.join(d, "config.json"), "wb") as f:
        f.write(b'{"model_type":"test","hidden_size":16}')
    with open(os.path.join(d, "tokenizer.json"), "wb") as f:
        f.write(b'{"version":"1.0"}')
    return d


def main() -> int:
    client = store.MooncakeDistributedStore()
    rc = client.setup(
        os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "localhost:17815"),
        os.environ.get("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE"),
        128 * 1024 * 1024, 64 * 1024 * 1024, "tcp", "", MASTER,
    )
    print("setup_ret", rc, flush=True)

    model_dir = _make_model_dir()
    # record ground-truth sha per file
    truth = {}
    for root, _, names in os.walk(model_dir):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), model_dir)
            with open(os.path.join(root, n), "rb") as f:
                truth[rel] = hashlib.sha256(f.read()).hexdigest()
    print("model files:", sorted(truth), flush=True)

    ident = WeightRevisionIdentity("default", "p1", "tiny-model", "v1", 1)
    fw = FileWeightStore(client, chunk_size=64)  # tiny chunk -> multi-chunk shards

    # --- import ---
    manifest = fw.put_model_dir(model_dir, ident, store)
    print(f"IMPORT OK: payload_count={manifest.payload_count} "
          f"logical_bytes={manifest.logical_bytes} files={len(manifest.files)}",
          flush=True)

    # --- open + verify every file byte-for-byte ---
    h = fw.open_model(ident)
    print("opened, availability:", h.view.metadata.availability, flush=True)
    bad = 0
    for rel in sorted(truth):
        data = h.read(rel)
        got = hashlib.sha256(data).hexdigest()
        ok = got == truth[rel]
        print(f"  [{'OK' if ok else 'MISMATCH'}] {rel} ({len(data)}B)", flush=True)
        if not ok:
            bad += 1
    if bad:
        print(f"P1_FAIL: {bad} file(s) mismatched", flush=True)
        return 1

    # --- delete ---
    dres = fw.delete(ident)
    print("delete result:", dres, flush=True)

    print("P1_PASS: import -> open -> sha256 verify -> delete all OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
