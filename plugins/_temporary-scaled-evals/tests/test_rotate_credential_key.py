# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext

import pytest

try:
    from scaled_evals.api import rotate_credential_key
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


class _Cursor:
    def __init__(self) -> None:
        self.rows = [
            {"id": "cred_one", "encrypted_payload": b"one"},
            {"id": "cred_two", "encrypted_payload": b"two"},
        ]
        self.updates: list[tuple[bytes, str]] = []

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *_args):  # noqa: ANN002, ANN204
        return None

    def execute(self, _query: str) -> None:
        return None

    def fetchall(self) -> list[dict[str, object]]:
        return self.rows

    def executemany(self, _query: str, values: list[tuple[bytes, str]]) -> None:
        self.updates.extend(values)


class _Connection:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *_args):  # noqa: ANN002, ANN204
        return None

    def transaction(self):  # noqa: ANN204
        return nullcontext()

    def cursor(self) -> _Cursor:
        return self._cursor


def test_rotate_credentials_updates_every_live_row(monkeypatch) -> None:  # noqa: ANN001
    cursor = _Cursor()
    monkeypatch.setattr(
        rotate_credential_key.psycopg,
        "connect",
        lambda *_args, **_kwargs: _Connection(cursor),
    )
    monkeypatch.setattr(rotate_credential_key.crypto, "reencrypt", lambda value: value + b"-new")

    assert rotate_credential_key.rotate_credentials() == 2
    assert cursor.updates == [(b"one-new", "cred_one"), (b"two-new", "cred_two")]


def test_rotate_credentials_dry_run_verifies_without_updates(monkeypatch) -> None:  # noqa: ANN001
    cursor = _Cursor()
    seen: list[bytes] = []
    monkeypatch.setattr(
        rotate_credential_key.psycopg,
        "connect",
        lambda *_args, **_kwargs: _Connection(cursor),
    )
    monkeypatch.setattr(
        rotate_credential_key.crypto,
        "reencrypt",
        lambda value: seen.append(value) or value,
    )

    assert rotate_credential_key.rotate_credentials(dry_run=True) == 2
    assert seen == [b"one", b"two"]
    assert cursor.updates == []
