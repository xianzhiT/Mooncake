"""File-level weight manifest: the METADATA payload describing a revision.

The authority (C++) treats this object as opaque bytes keyed by MakeWeightManifestKey
and validated only by sha256 + data_type==METADATA + group membership. Its *content*
is ours to define. This records the file inventory so open_model can reconstruct the
original directory, and so external SHA-256 acceptance is possible.

Schema: file-weight-manifest/v2
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field


@dataclass(frozen=True)
class ChunkEntry:
    key: str          # full Store object key of this chunk
    size: int         # byte length of this chunk


@dataclass(frozen=True)
class FileEntry:
    relpath: str              # model-dir-relative POSIX path
    role: str                 # "weight" (safetensors shard) | "aux" (config/tokenizer)
    size: int                 # total file byte size
    sha256: str               # sha256 of the whole file
    affinity_id: str          # residency affinity unit (all chunks of a file share it)
    chunks: list[ChunkEntry] = field(default_factory=list)


@dataclass(frozen=True)
class FileWeightManifest:
    identity: dict                      # the 5-tuple as a plain dict
    files: list[FileEntry]
    chunk_size: int
    payload_count: int                  # total chunk objects (NOT counting manifest)
    logical_bytes: int                  # sum of payload (chunk) sizes
    schema: str = "file-weight-manifest/v2"

    def to_bytes(self) -> bytes:
        """Deterministic JSON encoding (sorted keys) so sha256 is stable."""
        obj = {
            "schema": self.schema,
            "identity": self.identity,
            "chunk_size": self.chunk_size,
            "payload_count": self.payload_count,
            "logical_bytes": self.logical_bytes,
            "files": [asdict(f) for f in self.files],
        }
        return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def from_bytes(data: bytes) -> "FileWeightManifest":
        obj = json.loads(data.decode("utf-8"))
        files = [
            FileEntry(
                relpath=f["relpath"],
                role=f["role"],
                size=f["size"],
                sha256=f["sha256"],
                affinity_id=f["affinity_id"],
                chunks=[ChunkEntry(**c) for c in f["chunks"]],
            )
            for f in obj["files"]
        ]
        return FileWeightManifest(
            identity=obj["identity"],
            files=files,
            chunk_size=obj["chunk_size"],
            payload_count=obj["payload_count"],
            logical_bytes=obj["logical_bytes"],
            schema=obj.get("schema", "file-weight-manifest/v2"),
        )
