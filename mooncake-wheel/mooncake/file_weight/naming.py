"""Canonical key naming for file-level WeightStore.

Reproduces the C++ path/segment encoding so manifest keys and file payload
object keys line up with what the authority validates at commit time.

References (branch feature/file-level-weightstore @ 72862981):
  - EncodeWeightPathSegment / MakeWeightManifestKey
    mooncake-store/include/weight_management.h:491,510
"""

from __future__ import annotations

from .identity import WeightRevisionIdentity, make_payload_group_id

_HEX = "0123456789ABCDEF"


def encode_weight_path_segment(value: str) -> str:
    """Reproduce EncodeWeightPathSegment: RFC3986 unreserved-safe, %XX uppercase.

    Safe set: A-Z a-z 0-9 and - _ . ~  (everything else percent-encoded per
    UTF-8 byte). Matches C++ byte-wise encoding exactly.
    """
    out: list[str] = []
    for byte in value.encode("utf-8"):
        c = chr(byte)
        safe = (
            ("a" <= c <= "z")
            or ("A" <= c <= "Z")
            or ("0" <= c <= "9")
            or c in "-_.~"
        )
        if safe:
            out.append(c)
        else:
            out.append("%")
            out.append(_HEX[byte >> 4])
            out.append(_HEX[byte & 0x0F])
    return "".join(out)


def make_manifest_key(identity: WeightRevisionIdentity) -> str:
    """Reproduce MakeWeightManifestKey.

    weights/<ns>/<resource>/<revision>/<generation>/manifest
    (ns/resource/revision path-encoded; generation is a plain decimal).
    """
    return (
        "weights/"
        + encode_weight_path_segment(identity.namespace)
        + "/"
        + encode_weight_path_segment(identity.resource_id)
        + "/"
        + encode_weight_path_segment(identity.revision)
        + "/"
        + str(identity.weight_generation)
        + "/manifest"
    )


def make_file_object_key(
    identity: WeightRevisionIdentity, relpath: str, chunk_index: int
) -> str:
    """File payload object key. The key is free-form (authority only checks the
    object's group_id == canonical group), so we encode the model-relative path
    plus a chunk index. relpath segments are path-encoded for safety.

    Layout: <group>:file:<encoded-relpath>:chunk:<i>
    """
    group = make_payload_group_id(identity)
    enc = "/".join(encode_weight_path_segment(seg) for seg in relpath.split("/"))
    return f"{group}:file:{enc}:chunk:{chunk_index}"
