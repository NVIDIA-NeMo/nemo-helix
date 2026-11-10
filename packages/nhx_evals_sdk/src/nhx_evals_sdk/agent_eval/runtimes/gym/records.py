# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym's on-disk record format: the artifacts it writes, and the key that joins them.

Shared deliberately between :mod:`dataset` (which *writes* the index onto a materialized row) and
:mod:`results` (which *reads* it back off a rollout). The attribution contract is exactly this
pair of key names, so one definition means the writer and the reader cannot disagree.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


NG_TASK_INDEX = "_ng_task_index"
NG_ROLLOUT_INDEX = "_ng_rollout_index"
#: Gym appends this to a capture key past the first attempt, so it is part of the join.
NG_ATTEMPT_INDEX = "_ng_attempt_index"
#: Set on a record Gym routes to the failures sidecar instead of the rollouts file.
NG_FAILURE_CLASS = "_ng_failure_class"
#: Fields excluded from a row's content hash (runtime-injected, not task-defining).
RUNTIME_KEYS = frozenset({NG_TASK_INDEX, NG_ROLLOUT_INDEX})


#: Lines of subprocess output retained in memory for inclusion in a failure message.
def read_jsonl(path: str | Path, *, tolerant: bool = False) -> list[dict[str, Any]]:
    """Read a jsonl file. With ``tolerant=True``, skip (and log) malformed lines instead of raising —

    used for Gym's ``*_failures.jsonl`` sidecar, which is written during abnormal termination and can
    end in a truncated line; a corrupt failure record must not sink the successfully-collected trials.
    """
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                if not tolerant:
                    raise
                logger.warning("Skipping malformed JSON at %s:%d", path, line_no)
    return rows


def iter_jsonl_objects(
    source: str | Path | Iterable[bytes], *, name: str | None = None
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, row)`` for each non-blank line of a jsonl dataset, strictly.

    ``source`` is a path, or an iterable of byte lines (a binary file or ``io.BytesIO``) that
    ``name`` labels in errors. Unlike :func:`read_jsonl`, every failure names the physical line
    (blank lines counted), because a dataset author fixes a line, not a row position: invalid UTF-8,
    malformed JSON, and a line holding valid JSON that isn't an object (a scalar, array or ``null``)
    are all ``ValueError``.
    """
    if isinstance(source, (str, Path)):
        with Path(source).open("rb") as handle:
            yield from _jsonl_objects(handle, name or str(source))
    else:
        yield from _jsonl_objects(source, name or "<stream>")


def _jsonl_objects(lines: Iterable[bytes], label: str) -> Iterator[tuple[int, dict[str, Any]]]:
    for line_no, raw in enumerate(lines, 1):
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{label}:{line_no}: line is not valid UTF-8") from exc
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{line_no}: malformed JSON ({exc.msg})") from exc
        if not isinstance(value, dict):
            kind = "null" if value is None else type(value).__name__
            raise ValueError(f"{label}:{line_no}: expected a JSON object, got {kind}")
        yield line_no, value


#: `gym env start`'s combined output, under the run's work dir. Named here with the other
#: artifacts because a *collection* failure often has to point at it: the eval logs show the
#: symptom, this shows the cause.
ENV_LOG_NAME = "gym_env.log"
