"""Canonical identity, naming, and digest helpers for file-level WeightStore.

These reproduce the C++ authority's byte-exact algorithms so a pure-Python
file-level facade can talk to the managed weight RPCs directly.

References (branch feature/file-level-weightstore @ 72862981):
  - ComputeWeightIdentitySha256 / MakeWeightPayloadGroupId
    mooncake-store/src/weight_metadata_store.cpp
  - MakeWeightManifestKey / EncodeWeightPathSegment
    mooncake-store/include/weight_management.h

IMPORTANT: lengths and digests are computed over UTF-8 *bytes*, matching the
C++ std::string_view::size(). Do not use str length for multibyte values.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class WeightRevisionIdentity:
    """The 5-tuple identity. Mirrors C++ WeightRevisionIdentity field order."""

    tenant_id: str
    namespace: str
    resource_id: str
    revision: str
    weight_generation: int

    def __post_init__(self) -> None:
        for name in ("tenant_id", "namespace", "resource_id", "revision"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v:
                raise ValueError(f"{name} must be a non-empty string")
        if (
            not isinstance(self.weight_generation, int)
            or self.weight_generation < 1
            or self.weight_generation >= (1 << 64)
        ):
            raise ValueError("weight_generation must be a u64 >= 1")


def _append_length_prefixed(parts: list[bytes], value: str) -> None:
    """Append `{byte_len}:{value}\\n` as the C++ AppendLengthPrefixed does."""
    raw = value.encode("utf-8")
    parts.append(str(len(raw)).encode("ascii"))
    parts.append(b":")
    parts.append(raw)
    parts.append(b"\n")


def _sha256_hex(parts: list[bytes]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()  # lowercase hex, matches C++ kHex table


def compute_weight_identity_sha256(identity: WeightRevisionIdentity) -> str:
    """Reproduce ComputeWeightIdentitySha256 (5 length-prefixed fields)."""
    parts: list[bytes] = []
    _append_length_prefixed(parts, identity.tenant_id)
    _append_length_prefixed(parts, identity.namespace)
    _append_length_prefixed(parts, identity.resource_id)
    _append_length_prefixed(parts, identity.revision)
    _append_length_prefixed(parts, str(identity.weight_generation))
    return _sha256_hex(parts)


def make_payload_group_id(identity: WeightRevisionIdentity) -> str:
    """Reproduce MakeWeightPayloadGroupId: 'weight:' + identity sha256."""
    return "weight:" + compute_weight_identity_sha256(identity)


def compute_payload_keys_sha256(payload_keys: list[str]) -> str:
    """Reproduce ComputeWeightPayloadKeysSha256.

    Sort keys (byte order), then concatenate `{len}:{key}\\n` for each, sha256.
    Note: C++ std::sort on std::string is lexicographic over unsigned bytes;
    Python sorted() on the UTF-8-encoded bytes matches that. We sort on the
    encoded bytes to be exact for non-ASCII keys.
    """
    sorted_keys = sorted(payload_keys, key=lambda k: k.encode("utf-8"))
    parts: list[bytes] = []
    for key in sorted_keys:
        _append_length_prefixed(parts, key)
    return _sha256_hex(parts)
