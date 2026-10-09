"""File-level WeightStore facade (P0 scaffolding).

Pure-Python layer on top of the managed weight RPCs in `mooncake.store`.
See 权重管控面-社区方案适配/05 for the design.
"""

from .identity import (
    WeightRevisionIdentity,
    compute_weight_identity_sha256,
    make_payload_group_id,
    compute_payload_keys_sha256,
)
from .naming import (
    encode_weight_path_segment,
    make_manifest_key,
    make_file_object_key,
)

__all__ = [
    "WeightRevisionIdentity",
    "compute_weight_identity_sha256",
    "make_payload_group_id",
    "compute_payload_keys_sha256",
    "encode_weight_path_segment",
    "make_manifest_key",
    "make_file_object_key",
]
