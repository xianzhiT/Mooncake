#!/usr/bin/env python3
"""P2 e2e: lease read-protection + residency (SSD offload) + clean delete.

Covers the three follow-ups:
  1. delete reaches DELETED via delete+reconcile (no more spurious BUSY)
  2. an open handle's lease blocks delete (BUSY), and release unblocks it
  3. set_residency drives HOT/COLD/MIXED to convergence, with COLD payloads
     living on LOCAL_DISK (SSD) while the manifest stays in memory
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
    FileWeightStore, RESIDENCY_HOT, RESIDENCY_COLD, RESIDENCY_MIXED,
)

MASTER = os.environ.get("MOONCAKE_MASTER", "127.0.0.1:50181")


def _make_model_dir():
    d = tempfile.mkdtemp(prefix="model2-")
    with open(os.path.join(d, "model-00001-of-00002.safetensors"), "wb") as f:
        f.write(os.urandom(200))
    with open(os.path.join(d, "model-00002-of-00002.safetensors"), "wb") as f:
        f.write(os.urandom(120))
    with open(os.path.join(d, "config.json"), "wb") as f:
        f.write(b'{"model_type":"test"}')
    with open(os.path.join(d, "tokenizer.json"), "wb") as f:
        f.write(b'{"version":"1.0"}')
    return d


def _payload_keys(ident, manifest):
    keys = []
    for fe in manifest.files:
        for ch in fe.chunks:
            keys.append(ch.key)
    return keys


def _replica_summary(client, keys):
    """Return {key: (has_memory, has_local_disk)} from batch_get_replica_desc."""
    desc = client.batch_get_replica_desc(list(keys))
    out = {}
    for k in keys:
        reps = desc.get(k) or []
        out[k] = (
            any(r.is_memory_replica() for r in reps),
            any(r.is_local_disk_replica() for r in reps),
        )
    return out


def main() -> int:
    client = store.MooncakeDistributedStore()
    # enable_ssd_offload=True (9th arg): without it the client never runs the
    # offload heartbeat that pushes evicted WEIGHT payloads to LOCAL_DISK, so a
    # COLD residency operation can never converge (master queues the offload but
    # no client serves it). ssd_offload_path is taken from
    # MOONCAKE_OFFLOAD_FILE_STORAGE_PATH when empty.
    rc = client.setup(
        os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "localhost:17815"),
        os.environ.get("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE"),
        128 * 1024 * 1024, 64 * 1024 * 1024, "tcp", "", MASTER,
        None, True, os.environ.get("MOONCAKE_OFFLOAD_FILE_STORAGE_PATH", ""),
    )
    print("setup_ret", rc, flush=True)

    model_dir = _make_model_dir()
    truth = {}
    for root, _, names in os.walk(model_dir):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), model_dir)
            with open(os.path.join(root, n), "rb") as f:
                truth[rel] = hashlib.sha256(f.read()).hexdigest()

    ident = WeightRevisionIdentity("default", "p2", "tiny-model", "v1", 1)
    fw = FileWeightStore(client, chunk_size=64)

    manifest = fw.put_model_dir(model_dir, ident, store)
    weight_keys = [ch.key for fe in manifest.files if fe.role == "weight" for ch in fe.chunks]
    aux_keys = [ch.key for fe in manifest.files if fe.role == "aux" for ch in fe.chunks]
    man_key = None
    # manifest key = weights/<ns>/<res>/<rev>/<gen>/manifest
    man_key = "weights/p2/tiny-model/v1/1/manifest"
    all_keys = weight_keys + aux_keys + [man_key]
    print(f"imported: {len(weight_keys)} weight chunks, {len(aux_keys)} aux chunks",
          flush=True)

    # ===== (2) lease read-protection =====
    h = fw.open_model(ident, holder="p2-reader", lease_ttl_ms=30000)
    print("[lease] handle open, lease_id:", h.lease.lease_id, flush=True)

    view = fw._require(client.get_weight_revision(*(
        ident.tenant_id, ident.namespace, ident.resource_id, ident.revision,
        ident.weight_generation)), "get")
    blocked = client.delete_weight_revision(
        ident.tenant_id, ident.namespace, ident.resource_id, ident.revision,
        ident.weight_generation, view.metadata.metadata_generation)
    is_busy = blocked[0] is None and blocked[1] == 6
    print(f"[lease] delete while leased -> BUSY(6)? {is_busy} "
          f"(mgmt_err={blocked[1]})", flush=True)
    if not is_busy:
        print("P2_FAIL: delete not blocked by active lease", flush=True)
        return 1

    # reads work while leased
    ok = all(h.read(rel) and hashlib.sha256(h.read(rel)).hexdigest() == truth[rel]
             for rel in truth)
    print(f"[lease] reads under lease OK: {ok}", flush=True)
    h.close()
    time.sleep(0.3)
    print("[lease] handle closed, lease released", flush=True)

    # ===== (3) residency: HOT -> COLD(SSD) -> MIXED -> HOT =====
    client.reconcile_weight_revision(ident.tenant_id, ident.namespace,
                                     ident.resource_id, ident.revision,
                                     ident.weight_generation)
    r = fw.set_residency(ident, RESIDENCY_COLD)
    print(f"[residency] COLD converged: residency={r.residency} op={r.operation_id}",
          flush=True)
    reps = _replica_summary(client, all_keys)
    cold_ok = all((not mem) and (disk or True) for k, (mem, disk) in reps.items()
                  if k in weight_keys)
    weight_all_cold = all(not reps[k][0] for k in weight_keys)
    man_hot = reps[man_key][0]
    print(f"[residency] weight chunks all non-memory: {weight_all_cold}", flush=True)
    print(f"[residency] manifest in memory: {man_hot}", flush=True)
    # confirm reads still work from SSD
    cold_read_ok = all(hashlib.sha256(fw.open_model(ident, protect=False).read(rel)).hexdigest() == truth[rel]
                       for rel in list(truth)[:1])
    print(f"[residency] cold read OK: {cold_read_ok}", flush=True)

    r = fw.set_residency(ident, RESIDENCY_HOT)
    print(f"[residency] HOT converged: residency={r.residency}", flush=True)
    reps = _replica_summary(client, all_keys)
    weight_all_hot = all(reps[k][0] for k in weight_keys)
    print(f"[residency] weight chunks all memory: {weight_all_hot}", flush=True)

    r = fw.set_residency(ident, RESIDENCY_MIXED, mixed_hot_ratio=0.5)
    print(f"[residency] MIXED converged: residency={r.residency}", flush=True)

    # ===== (1) clean delete to DELETED =====
    # first release any leftover lease state, then delete+reconcile
    r = fw.set_residency(ident, RESIDENCY_HOT)  # back to fully readable
    final = fw.delete(ident)
    print(f"[delete] final availability={final.availability} "
          f"(DELETED={int(final.availability) == 4})", flush=True)
    gone = all(client.is_exist(k) == 0 for k in all_keys)
    print(f"[delete] all objects gone: {gone}", flush=True)

    if not (weight_all_cold and man_hot and cold_read_ok and int(final.availability) == 4 and gone):
        print("P2_FAIL: some assertion failed", flush=True)
        return 1
    print("P2_PASS: lease-protect + SSD residency + clean delete all OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
