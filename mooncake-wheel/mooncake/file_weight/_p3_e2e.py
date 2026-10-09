#!/usr/bin/env python3
"""P3 e2e: REAL-SCALE import/open/residency/delete.

P1/P2 used tiny synthetic bytes (chunk=64B). P3 exercises the production
regime that small files cannot surface:
  - 64 MiB chunk size (the manifestless convention / DEFAULT_CHUNK_SIZE)
  - multi-hundred-MB safetensors shards -> each spans several chunk objects
  - real SSD capacity on COLD offload, real bandwidth on zero-copy read-back

Validates, end to end against a real master:
  1. import a multi-shard model dir (64 MiB chunks, multi-chunk shards)
  2. zero-copy read every file back, SHA-256 byte-exact
  3. COLD -> weight payloads on LOCAL_DISK (SSD), manifest stays in memory,
     read still correct from SSD; HOT -> all back in memory
  4. clean delete to DELETED, all objects gone
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import time

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _SELF_DIR]

import store  # type: ignore

_SRC = "/src/mooncake-wheel/mooncake/file_weight"
_PKG_PARENT = tempfile.mkdtemp()
shutil.copytree(_SRC, os.path.join(_PKG_PARENT, "file_weight"))
sys.path.insert(0, _PKG_PARENT)
from file_weight import WeightRevisionIdentity  # noqa: E402
from file_weight.store import (  # noqa: E402
    FileWeightStore, DEFAULT_CHUNK_SIZE, RESIDENCY_HOT, RESIDENCY_COLD,
)

MASTER = os.environ.get("MOONCAKE_MASTER", "127.0.0.1:50181")

# Real-scale shard sizes (chosen to span multiple 64 MiB chunks each).
_MiB = 1024 * 1024
SHARD_A = 200 * _MiB   # -> 4 chunks (64+64+64+8)
SHARD_B = 150 * _MiB   # -> 3 chunks (64+64+22)


def _make_model_dir():
    d = tempfile.mkdtemp(prefix="model3-")
    # urandom is slow at these sizes; write in 16 MiB blocks.
    def _write(path, total):
        h = hashlib.sha256()
        with open(path, "wb") as f:
            remaining = total
            while remaining > 0:
                n = min(16 * _MiB, remaining)
                buf = os.urandom(n)
                f.write(buf)
                h.update(buf)
                remaining -= n
        return h.hexdigest()

    truth = {}
    truth["model-00001-of-00002.safetensors"] = _write(
        os.path.join(d, "model-00001-of-00002.safetensors"), SHARD_A)
    truth["model-00002-of-00002.safetensors"] = _write(
        os.path.join(d, "model-00002-of-00002.safetensors"), SHARD_B)
    with open(os.path.join(d, "config.json"), "wb") as f:
        f.write(b'{"model_type":"test","hidden_size":4096}')
    with open(os.path.join(d, "tokenizer.json"), "wb") as f:
        f.write(b'{"version":"1.0"}')
    for rel in ("config.json", "tokenizer.json"):
        with open(os.path.join(d, rel), "rb") as f:
            truth[rel] = hashlib.sha256(f.read()).hexdigest()
    return d, truth


def main() -> int:
    client = store.MooncakeDistributedStore()
    # Global segment must hold the whole model in memory for HOT. SHARD_A+B
    # ~= 350 MiB; give generous headroom. enable_ssd_offload=True so COLD can
    # offload to LOCAL_DISK (ssd path from MOONCAKE_OFFLOAD_FILE_STORAGE_PATH).
    rc = client.setup(
        os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "localhost:17815"),
        os.environ.get("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE"),
        2 * 1024 * _MiB, 256 * _MiB, "tcp", "", MASTER,
        None, True, os.environ.get("MOONCAKE_OFFLOAD_FILE_STORAGE_PATH", ""),
    )
    print("setup_ret", rc, flush=True)
    if rc != 0:
        return rc

    t0 = time.time()
    model_dir, truth = _make_model_dir()
    total_bytes = SHARD_A + SHARD_B
    print(f"model built: {total_bytes/_MiB:.0f} MiB in {time.time()-t0:.1f}s, "
          f"chunk={DEFAULT_CHUNK_SIZE//_MiB} MiB", flush=True)

    ident = WeightRevisionIdentity("default", "p3", "real-model", "v1", 1)
    fw = FileWeightStore(client)  # DEFAULT_CHUNK_SIZE = 64 MiB

    # ===== (1) import at real scale =====
    t0 = time.time()
    manifest = fw.put_model_dir(model_dir, ident, store)
    dt = time.time() - t0
    weight_keys = [ch.key for fe in manifest.files if fe.role == "weight"
                   for ch in fe.chunks]
    aux_keys = [ch.key for fe in manifest.files if fe.role == "aux"
                for ch in fe.chunks]
    man_key = "weights/p3/real-model/v1/1/manifest"
    all_keys = weight_keys + aux_keys + [man_key]
    print(f"[import] OK in {dt:.1f}s ({total_bytes/_MiB/dt:.0f} MiB/s): "
          f"payload_count={manifest.payload_count} "
          f"logical_bytes={manifest.logical_bytes} "
          f"weight_chunks={len(weight_keys)} aux_chunks={len(aux_keys)}",
          flush=True)
    # a 200 MiB shard at 64 MiB chunks must be 4 objects; 150 MiB -> 3
    multichunk_ok = len(weight_keys) == 7
    print(f"[import] multi-chunk sharding correct (7 weight chunks): "
          f"{multichunk_ok}", flush=True)

    # ===== (2) zero-copy read-back, byte-exact =====
    t0 = time.time()
    h = fw.open_model(ident, holder="p3-reader")
    bad = 0
    for rel in sorted(truth):
        got = hashlib.sha256(h.read(rel)).hexdigest()
        ok = got == truth[rel]
        if not ok:
            bad += 1
            print(f"  [MISMATCH] {rel}", flush=True)
    dt = time.time() - t0
    print(f"[read] all {len(truth)} files SHA-256 OK: {bad == 0} "
          f"({total_bytes/_MiB/dt:.0f} MiB/s incl lease)", flush=True)
    h.close()
    time.sleep(0.3)

    # ===== (3) COLD -> SSD, then HOT =====
    t0 = time.time()
    r = fw.set_residency(ident, RESIDENCY_COLD, timeout_s=120.0)
    print(f"[cold] converged in {time.time()-t0:.1f}s residency={r.residency}",
          flush=True)
    desc = client.batch_get_replica_desc(all_keys)
    weight_all_cold = all(
        not any(x.is_memory_replica() for x in (desc.get(k) or []))
        and any(x.is_local_disk_replica() for x in (desc.get(k) or []))
        for k in weight_keys)
    man_hot = any(x.is_memory_replica() for x in (desc.get(man_key) or []))
    print(f"[cold] weight chunks all on SSD (non-mem + local_disk): "
          f"{weight_all_cold}", flush=True)
    print(f"[cold] manifest stays in memory: {man_hot}", flush=True)
    # read back from SSD, byte-exact, for the big shard
    big = "model-00001-of-00002.safetensors"
    h2 = fw.open_model(ident, protect=False)
    cold_read_ok = hashlib.sha256(h2.read(big)).hexdigest() == truth[big]
    print(f"[cold] SSD read-back of {SHARD_A//_MiB} MiB shard OK: "
          f"{cold_read_ok}", flush=True)

    r = fw.set_residency(ident, RESIDENCY_HOT, timeout_s=120.0)
    desc = client.batch_get_replica_desc(all_keys)
    weight_all_hot = all(
        any(x.is_memory_replica() for x in (desc.get(k) or []))
        for k in weight_keys)
    print(f"[hot] back in memory: {weight_all_hot} residency={r.residency}",
          flush=True)

    # ===== (4) clean delete =====
    final = fw.delete(ident, timeout_s=120.0)
    gone = all(client.is_exist(k) == 0 for k in all_keys)
    print(f"[delete] availability={final.availability} "
          f"(DELETED={int(final.availability) == 4}) all_gone={gone}",
          flush=True)

    if not (multichunk_ok and bad == 0 and weight_all_cold and man_hot
            and cold_read_ok and weight_all_hot
            and int(final.availability) == 4 and gone):
        print("P3_FAIL: some assertion failed", flush=True)
        return 1
    print("P3_PASS: real-scale import/read/SSD-residency/delete all OK",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
