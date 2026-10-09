"""FileWeightStore — file-level facade over the managed weight RPCs.

Pure Python on top of mooncake.store.MooncakeDistributedStore. Imports a model
directory as one managed weight revision (atomic group), loads it back (zero-copy
reads from Store replicas), and deletes it. See 权重管控面-社区方案适配/05.

Design invariants enforced here (verified against C++ commit validation,
weight_store_manager.cpp:193-249):
  - payload object: data_type=WEIGHT, group_ids=[canonical], residency_affinity_ids=[file]
  - manifest object: data_type=METADATA, group_ids=[canonical]
  - payload_count = number of chunk objects (manifest NOT counted)
  - logical_bytes = sum of chunk sizes
  - payload_keys_sha256 over the chunk keys (sorted, length-prefixed)
  - affinity_count = number of distinct affinity ids (== file count)
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass

from .identity import (
    WeightRevisionIdentity,
    make_payload_group_id,
    compute_payload_keys_sha256,
)
from .naming import make_manifest_key, make_file_object_key
from .manifest import FileWeightManifest, FileEntry, ChunkEntry

DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024  # 64 MiB, matches manifestless convention

# Residency enum ints (mirror store.WeightResidencyState / mooncake C++)
RESIDENCY_HOT = 1
RESIDENCY_COLD = 2
RESIDENCY_MIXED = 3
MIGRATION_MANUAL = 1
MIGRATION_AUTO = 2

# Non-weight auxiliary files kept always-HOT (never offloaded to SSD).
_AUX_SUFFIXES = (".json", ".txt", ".model", ".tokenizer", ".vocab", ".merges")


def _is_weight_file(relpath: str) -> bool:
    return relpath.endswith(".safetensors") or relpath.endswith(".bin")


def _affinity_for(relpath: str) -> str:
    # one affinity unit per file; stable id derived from the path
    return "aff-" + hashlib.sha1(relpath.encode("utf-8")).hexdigest()[:16]


@dataclass
class _PlannedFile:
    relpath: str
    role: str
    size: int
    sha256: str
    affinity_id: str
    chunk_keys: list  # list[(key, data_or_offset, size)]


class FileWeightError(RuntimeError):
    pass


class FileWeightStore:
    def __init__(self, client, chunk_size: int = DEFAULT_CHUNK_SIZE):
        """client: a connected store.MooncakeDistributedStore; store_mod: the store module."""
        self._c = client
        self._chunk = chunk_size

    # ---- helpers ---------------------------------------------------------
    def _require(self, result, op):
        value, mgmt_err, transport_err = result
        if value is None or mgmt_err != 0 or transport_err != 0:
            raise FileWeightError(
                f"{op} failed: mgmt_err={mgmt_err} transport_err={transport_err}"
            )
        return value

    def _scan(self, local_dir: str):
        files = []
        for root, _, names in os.walk(local_dir):
            for n in sorted(names):
                full = os.path.join(root, n)
                rel = os.path.relpath(full, local_dir).replace(os.sep, "/")
                files.append((rel, full))
        return sorted(files)

    # ---- import ----------------------------------------------------------
    def put_model_dir(self, local_dir: str, identity: WeightRevisionIdentity,
                      store_mod, policy: dict | None = None) -> FileWeightManifest:
        """Import a model directory as one managed weight revision (HOT)."""
        if identity.tenant_id != "default":
            raise FileWeightError("tenant must be 'default' (multi-tenancy off)")
        group = make_payload_group_id(identity)
        mkey = make_manifest_key(identity)

        # 1) plan: scan + chunk + affinity
        planned: list[_PlannedFile] = []
        payload_count = 0
        logical_bytes = 0
        for rel, full in self._scan(local_dir):
            size = os.path.getsize(full)
            role = "weight" if _is_weight_file(rel) else "aux"
            aff = _affinity_for(rel)
            h = hashlib.sha256()
            chunks = []
            with open(full, "rb") as fh:
                idx = 0
                while True:
                    buf = fh.read(self._chunk)
                    if not buf:
                        break
                    h.update(buf)
                    key = make_file_object_key(identity, rel, idx)
                    chunks.append((key, buf, len(buf)))
                    payload_count += 1
                    logical_bytes += len(buf)
                    idx += 1
            if not chunks:  # zero-byte file: still needs one (empty) chunk object
                key = make_file_object_key(identity, rel, 0)
                chunks.append((key, b"", 0))
                payload_count += 1
            planned.append(_PlannedFile(rel, role, size, h.hexdigest(), aff, chunks))

        affinity_count = len(planned)
        if affinity_count == 0:
            raise FileWeightError(f"no files found under {local_dir}")

        # 2) begin_weight_import (empty group -> server fills canonical)
        # Always send an explicit policy. The C++ default is MIXED+AUTO
        # (weight_management.h:188-190); inheriting it makes CommitImport
        # auto-spawn a MIGRATING operation (weight_metadata_store.cpp:357-382),
        # whose leftover operation_id then blocks a later update_weight_policy
        # with BUSY (weight_metadata_store.cpp:537). We import HOT+MANUAL so the
        # revision lands quiescent and set_residency() controls migration.
        pol = policy or {}
        pref = pol.get("preferred_residency", RESIDENCY_HOT)
        ratio = pol.get("mixed_hot_ratio", 0.5)
        mig = pol.get("migration_mode", MIGRATION_MANUAL)
        has_policy = True
        affinity_digest = hashlib.sha256(
            ",".join(p.affinity_id for p in planned).encode("utf-8")
        ).hexdigest()
        importing = self._require(
            self._c.begin_weight_import(
                identity.tenant_id, identity.namespace, identity.resource_id,
                identity.revision, identity.weight_generation,
                "", payload_count, logical_bytes,
                has_policy, int(pref), float(ratio), int(mig),
                affinity_count, affinity_digest,
            ),
            "begin_weight_import",
        )
        gen = importing.metadata_generation
        assert importing.manifest.payload_group_id == group, "group_id mismatch"

        try:
            # 3) put all payload chunks (WEIGHT + group + affinity + hard_pin)
            all_payload_keys = []
            for pf in planned:
                for (key, data, _sz) in pf.chunk_keys:
                    cfg = store_mod.ReplicateConfig()
                    cfg.replica_num = 1
                    cfg.with_hard_pin = True
                    cfg.group_ids = [group]
                    cfg.residency_affinity_ids = [pf.affinity_id]
                    cfg.data_type = store_mod.ObjectDataType.WEIGHT
                    if self._c.put(key, data, cfg) != 0:
                        raise FileWeightError(f"put payload failed: {key}")
                    all_payload_keys.append(key)

            # 4) build + put manifest (METADATA)
            manifest = FileWeightManifest(
                identity={
                    "tenant_id": identity.tenant_id,
                    "namespace": identity.namespace,
                    "resource_id": identity.resource_id,
                    "revision": identity.revision,
                    "weight_generation": identity.weight_generation,
                },
                files=[
                    FileEntry(
                        relpath=pf.relpath, role=pf.role, size=pf.size,
                        sha256=pf.sha256, affinity_id=pf.affinity_id,
                        chunks=[ChunkEntry(key=k, size=s) for (k, _d, s) in pf.chunk_keys],
                    )
                    for pf in planned
                ],
                chunk_size=self._chunk,
                payload_count=payload_count,
                logical_bytes=logical_bytes,
            )
            manifest_bytes = manifest.to_bytes()
            mcfg = store_mod.ReplicateConfig()
            mcfg.replica_num = 1
            mcfg.with_hard_pin = True
            mcfg.group_ids = [group]
            mcfg.data_type = store_mod.ObjectDataType.METADATA
            if self._c.put(mkey, manifest_bytes, mcfg) != 0:
                raise FileWeightError(f"put manifest failed: {mkey}")

            # 5) commit
            ready = self._require(
                self._c.commit_weight_import(
                    identity.tenant_id, identity.namespace, identity.resource_id,
                    identity.revision, identity.weight_generation,
                    gen,
                    mkey,
                    hashlib.sha256(manifest_bytes).hexdigest(),
                    group,
                    compute_payload_keys_sha256(all_payload_keys),
                    payload_count,
                    logical_bytes,
                ),
                "commit_weight_import",
            )
            if ready.availability != store_mod.WeightAvailabilityState.READY:
                raise FileWeightError(f"not READY after commit: {ready.availability}")
            return manifest
        except Exception:
            # best-effort abort; leaves no READY revision
            try:
                self._c.abort_weight_import(
                    identity.tenant_id, identity.namespace, identity.resource_id,
                    identity.revision, identity.weight_generation, gen,
                )
            except Exception:
                pass
            raise

    # ---- load (zero-copy read, lease-protected) --------------------------
    def open_model(self, identity: WeightRevisionIdentity, holder: str = "file_weight",
                   lease_ttl_ms: int = 30_000, protect: bool = True):
        """Return a handle exposing per-file reads straight from Store.

        When protect=True (default), acquires a generation-fenced revision lease
        and keeps it renewed in the background, so the revision cannot be deleted
        or have its readable replicas released while the handle is open. The
        caller must close() the handle to release the lease.
        """
        view = self._require(
            self._c.get_weight_revision(
                identity.tenant_id, identity.namespace, identity.resource_id,
                identity.revision, identity.weight_generation,
            ),
            "get_weight_revision",
        )
        lease = None
        if protect:
            lease = self._require(
                self._c.acquire_weight_revision_lease(
                    identity.tenant_id, identity.namespace, identity.resource_id,
                    identity.revision, identity.weight_generation,
                    view.metadata.metadata_generation, holder, lease_ttl_ms,
                ),
                "acquire_weight_revision_lease",
            )
        mkey = make_manifest_key(identity)
        manifest = FileWeightManifest.from_bytes(self._c.get(mkey))
        return FileWeightHandle(self._c, identity, manifest, view, lease,
                                lease_ttl_ms)

    # ---- delete (delete + reconcile to DELETED) --------------------------
    def delete(self, identity: WeightRevisionIdentity, timeout_s: float = 60.0):
        """Delete a revision and drive reconciliation until it reaches DELETED.

        The first delete_weight_revision retires leases and removes one batch,
        leaving the revision in DELETING; reconcile_weight_revision advances the
        rest. Returns the final metadata. Raises FileWeightError on BUSY (an
        active lease still held) or timeout.
        """
        view = self._require(
            self._c.get_weight_revision(
                identity.tenant_id, identity.namespace, identity.resource_id,
                identity.revision, identity.weight_generation,
            ),
            "get_weight_revision",
        )
        first = self._c.delete_weight_revision(
            identity.tenant_id, identity.namespace, identity.resource_id,
            identity.revision, identity.weight_generation,
            view.metadata.metadata_generation,
        )
        # BUSY here means a lease/upsert is still active -> surface it.
        if first[1] == 6:  # WeightManagementError.BUSY
            raise FileWeightError("delete BUSY: revision still has an active lease")
        if first[1] not in (0,) and first[0] is None:
            raise FileWeightError(f"delete failed: mgmt_err={first[1]}")
        deadline = time.monotonic() + timeout_s
        last = None
        while time.monotonic() < deadline:
            res = self._require(
                self._c.reconcile_weight_revision(
                    identity.tenant_id, identity.namespace, identity.resource_id,
                    identity.revision, identity.weight_generation,
                ),
                "reconcile_weight_revision",
            )
            last = res
            if int(res.availability) == 4:  # WeightAvailabilityState.DELETED
                return res
            time.sleep(0.1)
        raise FileWeightError(f"delete did not reach DELETED within {timeout_s}s "
                              f"(last availability={getattr(last, 'availability', None)})")

    # ---- residency (HOT/COLD/MIXED, SSD offload) -------------------------
    def set_residency(self, identity: WeightRevisionIdentity, target: int,
                      mixed_hot_ratio: float | None = None,
                      migration_mode: int = MIGRATION_MANUAL,
                      timeout_s: float = 60.0):
        """Move a revision to a target residency and drive it to convergence.

        target: RESIDENCY_HOT / RESIDENCY_COLD / RESIDENCY_MIXED.
        COLD offloads weight payloads to LOCAL_DISK (SSD). Convergence is
        advanced via reconcile_weight_revision (the master also reconciles in
        the background, but we drive it explicitly and wait).
        """
        cur = self._require(
            self._c.get_weight_revision(
                identity.tenant_id, identity.namespace, identity.resource_id,
                identity.revision, identity.weight_generation,
            ),
            "get_weight_revision",
        )
        updated = self._require(
            self._c.update_weight_policy(
                identity.tenant_id, identity.namespace, identity.resource_id,
                identity.revision, identity.weight_generation,
                cur.metadata.metadata_generation, int(target),
                float(mixed_hot_ratio) if mixed_hot_ratio is not None else 0.5,
                int(migration_mode),
            ),
            "update_weight_policy",
        )
        op = self._require(
            self._c.start_weight_residency_operation(
                identity.tenant_id, identity.namespace, identity.resource_id,
                identity.revision, identity.weight_generation,
                updated.metadata_generation, int(target),
                float(mixed_hot_ratio) if mixed_hot_ratio is not None else None,
            ),
            "start_weight_residency_operation",
        )
        deadline = time.monotonic() + timeout_s
        last_res = None
        while time.monotonic() < deadline:
            res = self._require(
                self._c.reconcile_weight_revision(
                    identity.tenant_id, identity.namespace, identity.resource_id,
                    identity.revision, identity.weight_generation,
                ),
                "reconcile_weight_revision",
            )
            last_res = res
            if int(res.residency) == int(target) and res.operation_id is None:
                return res
            time.sleep(0.1)
        raise FileWeightError(
            f"residency did not converge to {target} in {timeout_s}s "
            f"(residency={getattr(last_res, 'residency', None)} "
            f"op={getattr(last_res, 'operation_id', None)} "
            f"avail={getattr(last_res, 'availability', None)} "
            f"hot_ratio={getattr(last_res, 'observed_hot_ratio', None)})")


class FileWeightHandle:
    def __init__(self, client, identity, manifest: FileWeightManifest, view,
                 lease=None, lease_ttl_ms: int = 30_000):
        self._c = client
        self.identity = identity
        self.manifest = manifest
        self.view = view
        self.lease = lease
        self._lease_ttl_ms = lease_ttl_ms
        self._closed = False
        self._renew_stop = threading.Event()
        self._renew_thread = None
        self._by_rel = {f.relpath: f for f in manifest.files}
        if lease is not None:
            self._renew_thread = threading.Thread(
                target=self._renew_loop, daemon=True
            )
            self._renew_thread.start()

    def _renew_loop(self):
        interval = max(self._lease_ttl_ms / 3000.0, 1.0)  # renew ~1/3 of TTL
        while not self._renew_stop.wait(interval):
            try:
                res = self._c.renew_weight_revision_lease(
                    self.identity.tenant_id, self.lease.lease_id,
                    self._lease_ttl_ms,
                )
                if res[1] != 0 and res[2] != 0:
                    break
            except Exception:
                break

    def files(self):
        return list(self._by_rel.keys())

    def read(self, relpath: str) -> bytes:
        """Reassemble a file by concatenating its chunk objects (zero-copy from Store)."""
        f = self._by_rel.get(relpath)
        if f is None:
            raise FileWeightError(f"no such file in revision: {relpath}")
        out = bytearray()
        for ch in f.chunks:
            out += self._c.get(ch.key)
        return bytes(out)

    def close(self):
        """Release the lease (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._renew_stop.set()
        if self._renew_thread is not None:
            self._renew_thread.join(timeout=2)
        if self.lease is not None:
            try:
                self._c.release_weight_revision_lease(
                    self.identity.tenant_id, self.lease.lease_id
                )
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
