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
    pass


class CatalogOwnershipLostError(RuntimeError):
    pass


class CatalogOwnershipReleaseUnknownError(RuntimeError):
    pass


@dataclass(frozen=True)
class MutationOwnership:
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
        ownership = self._acquire(operation, checkpoint_id)
        try:
            yield ownership
        finally:
            self._release(ownership.token)

    def _acquire(self, operation: str, checkpoint_id: str) -> MutationOwnership:
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
                return ownership
            if observed is not None:
                current = observed
                read_error = None
            elif error is not None:
                read_error = error

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CatalogMutationBusyError(
                    self._busy_message(operation, checkpoint_id, current, read_error)
                )
            time.sleep(min(self.poll_interval_seconds, remaining))

    def _release(self, token: str) -> None:
        # This check prevents an obvious wrong-owner remove, but it is not a
        # conditional delete and therefore is not fencing.
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
