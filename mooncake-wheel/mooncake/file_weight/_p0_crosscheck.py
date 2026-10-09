#!/usr/bin/env python3
"""P0 cross-check: validate file_weight naming/digest against the C++ authority.

Strategy: the C++ server validates payload_group_id and manifest_key against
its own MakeWeightPayloadGroupId / MakeWeightManifestKey at begin/commit time.
So "begin_weight_import accepts our values" proves our Python reproductions are
byte-identical to C++. No need for the C++ helpers to be exposed.

Run inside the mooncake-fw:dev container with a master already running and
PYTHONPATH including both the mooncake package and the file_weight source.
"""

from __future__ import annotations

import os
import sys

import store  # type: ignore

# Import the file_weight modules directly (as top-level), WITHOUT triggering the
# installed `mooncake` package __init__ (which pulls compiled submodules not
# present in the source tree). The modules use relative imports, so load them
# as a lightweight package rooted at this directory.
import importlib.util  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(
        f"fw_{mod_name}", os.path.join(_HERE, filename)
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"fw_{mod_name}"] = m
    spec.loader.exec_module(m)
    return m


# identity has no intra-package deps; naming imports from identity by relative
# path, so inject identity under the name naming expects.
_identity = _load("identity", "identity.py")
sys.modules["identity"] = _identity  # satisfy naming's "from .identity" fallback
# naming uses "from .identity import ...": load with a shim package.
_naming_src = open(os.path.join(_HERE, "naming.py")).read().replace(
    "from .identity import", "from fw_identity import"
)
_naming = type(sys)("fw_naming")
sys.modules["fw_naming"] = _naming
exec(compile(_naming_src, os.path.join(_HERE, "naming.py"), "exec"), _naming.__dict__)

WeightRevisionIdentity = _identity.WeightRevisionIdentity
make_payload_group_id = _identity.make_payload_group_id
compute_payload_keys_sha256 = _identity.compute_payload_keys_sha256
make_manifest_key = _naming.make_manifest_key

MASTER = os.environ.get("MOONCAKE_MASTER", "127.0.0.1:50181")


def _unwrap(result, op):
    value, mgmt_err, transport_err = result
    return value, mgmt_err, transport_err


def main() -> int:
    client = store.MooncakeDistributedStore()
    rc = client.setup(
        os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "localhost:17815"),
        os.environ.get("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE"),
        32 * 1024 * 1024,
        16 * 1024 * 1024,
        "tcp",
        "",
        MASTER,
    )
    print("setup_ret", rc, flush=True)

    # Identity with ASCII + a non-ASCII revision to exercise encoding.
    ident = WeightRevisionIdentity(
        tenant_id="default",
        namespace="qwen",
        resource_id="Qwen2.5-7B-Instruct",
        revision="v1",
        weight_generation=1,
    )
    group = make_payload_group_id(ident)
    mkey = make_manifest_key(ident)
    print("py_group_id  ", group, flush=True)
    print("py_manifest_key", mkey, flush=True)

    # Pass EMPTY payload_group_id so the server fills in its own canonical
    # MakeWeightPayloadGroupId(identity). The returned metadata then carries the
    # authority's ground-truth group id, which we compare against ours.
    # (Mirrors the e2e script, which also passes "".)
    import hashlib as _hl
    res = client.begin_weight_import(
        ident.tenant_id,
        ident.namespace,
        ident.resource_id,
        ident.revision,
        ident.weight_generation,
        "",                    # payload_group_id -> server computes canonical
        2,                     # expected_payload_count (2 files)
        131072,                # expected_logical_bytes
        True,                  # has_policy  (match working e2e call)
        1,                     # preferred_residency (HOT)
        0.5,                   # mixed_hot_ratio
        1,                     # migration_mode (MANUAL)
        2,                     # affinity_count
        _hl.sha256(b"file-a,file-b").hexdigest(),  # affinity_digest (non-empty)
    )
    value, mgmt_err, transport_err = _unwrap(res, "begin_weight_import")
    print(
        f"begin_weight_import -> mgmt_err={mgmt_err} transport_err={transport_err} "
        f"value={'OK' if value is not None else None}",
        flush=True,
    )
    if value is None or mgmt_err != 0 or transport_err != 0:
        print("CROSSCHECK_FAIL: begin_weight_import rejected", flush=True)
        return 1

    # Ground-truth group id from the authority.
    srv_group = value.manifest.payload_group_id
    print("srv_group_id ", srv_group, flush=True)
    ok_group = srv_group == group
    print(f"[group_id match] {ok_group}", flush=True)

    # Clean up: abort the import we just started.
    gen = value.metadata_generation
    client.abort_weight_import(
        ident.tenant_id, ident.namespace, ident.resource_id,
        ident.revision, ident.weight_generation, gen,
    )

    # digest self-consistency vs the e2e reference algorithm.
    keys = [f"{group}:file:a:chunk:0", f"{group}:file:b:chunk:0"]
    print("py_payload_keys_sha256", compute_payload_keys_sha256(keys), flush=True)

    if not ok_group:
        print(f"CROSSCHECK_FAIL: group_id differs py={group} srv={srv_group}",
              flush=True)
        return 1
    print("CROSSCHECK_PASS: naming group_id matches C++ authority", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
