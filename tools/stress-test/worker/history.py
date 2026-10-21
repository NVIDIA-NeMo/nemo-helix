# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSONL history written by a stress job. Credentials are dropped before the write."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

FLUSH_SECONDS = 30.0
_BLOCKED = {"token", "secret", "password", "authorization", "access_key", "value"}


def classify(service: str, operation: str, status: int) -> str:
    """A secret-value denial is an expected sample. Every other status is a result."""
    if service == "secrets" and operation == "value_read" and status == 403:
        return "expected_denial"
    return "result"


def scrub(record: dict[str, object]) -> str:
    safe = {key: value for key, value in record.items() if key not in _BLOCKED}
    return json.dumps(safe, sort_keys=True)


class HistoryLog:
    """Append-only log that flushes on a timer and again on close."""

    def __init__(self, path: Path, clock: Callable[[], float] | None = None) -> None:
        self.path = path
        self._clock = clock or time.monotonic
        self._secrets: list[str] = []
        self._last = self._clock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")

    def deny(self, secret: str) -> None:
        if secret:
            self._secrets.append(secret)

    def write(self, record: dict[str, object]) -> None:
        text = scrub(record)
        for secret in self._secrets:
            text = text.replace(secret, "[redacted]")
        self._handle.write(text + "\n")
        if self._clock() - self._last >= FLUSH_SECONDS:
            self._handle.flush()
            self._last = self._clock()

    def close(self) -> None:
        self._handle.flush()
        self._handle.close()
