"""Phase 1 coordination for weight-catalog mutations.

A model import or delete updates several Store objects, so object-level Put
semantics alone cannot make the whole catalog change atomic. Phase 1 therefore
serializes every catalog mutation through one Store-backed ownership record:

1. A writer uses ordinary, non-overwriting Put to publish a unique token.
2. It reads the record back; only the writer that sees its own token proceeds.
3. It keeps ownership for the complete import or delete operation.
4. Before release, it verifies the token again and then removes the record.

This is deliberately fail-closed. The record is hard-pinned against normal
eviction, but it is not a lease, transaction, conditional delete, or fencing
mechanism. Phase 1 therefore does not automatically take over after a writer or
Store failure.
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from .model_keyspace import catalog_mutation_owner_key


class CatalogMutationBusyError(RuntimeError):
    """Raised when another writer owns the catalog or ownership is uncertain."""

    pass


class CatalogOwnershipLostError(RuntimeError):
    """Raised when a writer can no longer prove that it owns the catalog."""

    pass


class CatalogOwnershipReleaseUnknownError(RuntimeError):
    """Raised when Store cannot confirm whether ownership removal succeeded."""

    pass


@dataclass(frozen=True)
class MutationOwnership:
    """Identity and diagnostics persisted in the global ownership record."""

    token: str
    operation: str
    checkpoint_id: str
    hostname: str
    pid: int
    started_at: int
    schema_version: int = 1

    def to_json_bytes(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> MutationOwnership:
        return cls(**json.loads(payload.decode("utf-8")))


class StoreMutationCoordinator:
    """Serializes catalog writes across processes and hosts through Store.

    ``config`` must describe hard-pinned metadata. Correctness also relies on
    ordinary Store Put being first-writer-wins: an existing ownership record
    must not be overwritten. Because Store reports an existing object as a
    successful Put, acquisition always uses read-back token verification.
    """

    def __init__(
        self,
        store,
        config,
        *,
        wait_timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.1,
    ) -> None:
        if wait_timeout_seconds < 0:
            raise ValueError("wait_timeout_seconds must not be negative")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self.store = store
        self.config = config
        self.wait_timeout_seconds = wait_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.key = catalog_mutation_owner_key()

    @contextmanager
    def mutation(
        self, operation: str, checkpoint_id: str
    ) -> Iterator[MutationOwnership]:
        """Hold global catalog ownership for one complete mutation."""

        ownership = self._acquire(operation, checkpoint_id)
        try:
            yield ownership
        finally:
            self._release(ownership.token)

    def _acquire(self, operation: str, checkpoint_id: str) -> MutationOwnership:
        # A fresh token distinguishes this attempt from every concurrent or
        # previously crashed writer. The remaining fields are diagnostic only;
        # they are never used to infer that an owner is dead.
        ownership = MutationOwnership(
            token=uuid.uuid4().hex,
            operation=operation,
            checkpoint_id=checkpoint_id,
            hostname=socket.gethostname(),
            pid=os.getpid(),
            started_at=int(time.time()),
        )
        payload = ownership.to_json_bytes()
        deadline = time.monotonic() + self.wait_timeout_seconds
        current: MutationOwnership | None = None
        read_error: str | None = None

        while True:
            # Store Put is first-writer-wins, but OBJECT_ALREADY_EXISTS is
            # reported as success by the client. Read-back identifies which
            # token actually won without using an overwriting upsert.
            try:
                result = self.store.put(self.key, payload, self.config)
                if result not in (0, None):
                    read_error = f"put returned {result}"
            # Store bindings may raise backend-specific exception types.
            except Exception as exc:  # noqa: BLE001
                read_error = f"put failed: {exc!r}"

            observed, error = self._read_ownership()
            if observed is not None and observed.token == ownership.token:
                # Put plus read-back is the Phase 1 claim protocol. Returning
                # before this equality check would let two writers proceed when
                # Store maps OBJECT_ALREADY_EXISTS to success.
                return ownership
            if observed is not None:
                # A different token is authoritative even if our Put returned
                # success: that writer won the first-write race.
                current = observed
                read_error = None
            elif error is not None:
                # Missing, malformed, and failed reads are all uncertain. Wait
                # until the deadline, but never enter the critical section.
                read_error = error

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CatalogMutationBusyError(
                    self._busy_message(operation, checkpoint_id, current, read_error)
                )
            time.sleep(min(self.poll_interval_seconds, remaining))

    def _release(self, token: str) -> None:
        # Read-before-remove prevents an obvious wrong-owner delete. It is not
        # atomic with remove, so it is only defensive validation, not fencing.
        # Automatic takeover is intentionally forbidden while that gap exists.
        current, error = self._read_ownership()
        if current is None:
            detail = error or "ownership record is missing"
            raise CatalogOwnershipLostError(
                f"cannot release catalog mutation ownership: {detail}"
            )
        if current.token != token:
            raise CatalogOwnershipLostError(
                "cannot release catalog mutation ownership: token changed to "
                f"{current.token}"
            )
        try:
            try:
                result = self.store.remove(self.key, True)
            except TypeError:
                result = self.store.remove(self.key)
        except Exception as exc:
            # A transport error does not reveal whether Store applied remove.
            # Report the outcome as unknown rather than claiming the lock is
            # held or released and allowing unsafe recovery.
            raise CatalogOwnershipReleaseUnknownError(
                "catalog mutation ownership release outcome is unknown: "
                f"remove failed: {exc!r}"
            ) from exc
        if result not in (0, None):
            raise CatalogOwnershipReleaseUnknownError(
                "catalog mutation ownership release outcome is unknown: "
                f"remove returned {result}"
            )

    def _read_ownership(
        self,
    ) -> tuple[MutationOwnership | None, str | None]:
        """Read and validate ownership without converting uncertainty to absence."""

        try:
            value = self.store.get(self.key)
        # Store bindings may raise backend-specific exception types.
        except Exception as exc:  # noqa: BLE001
            return None, f"ownership read failed: {exc!r}"
        if value is None:
            return None, "ownership record is not readable"
        try:
            payload = value.encode("utf-8") if isinstance(value, str) else bytes(value)
            return MutationOwnership.from_json_bytes(payload), None
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            return None, f"ownership record is invalid: {exc!r}"

    @staticmethod
    def _busy_message(
        operation: str,
        checkpoint_id: str,
        current: MutationOwnership | None,
        read_error: str | None,
    ) -> str:
        prefix = (
            f"catalog mutation is busy; cannot {operation} checkpoint {checkpoint_id!r}"
        )
        if current is not None:
            return (
                f"{prefix}; owner={current.operation} "
                f"checkpoint={current.checkpoint_id!r} "
                f"host={current.hostname} pid={current.pid} "
                f"started_at={current.started_at}"
            )
        return f"{prefix}; {read_error or 'owner is unknown'}"
