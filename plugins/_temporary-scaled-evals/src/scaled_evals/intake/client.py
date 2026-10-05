# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP client for NeMo Helix Intake v2 ATIF ingest.

Posts to ``/apis/intake/v2/workspaces/{workspace}/ingest/atif``. No
``Authorization`` header is sent, so this only reaches Intake deployments that
do not require one; targeting an authenticated endpoint means adding a
credential here first.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from scaled_evals.intake.atif_payload import IntakeError, TrialPayload

LOG = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 1.0
RETRYABLE_HTTP_STATUSES = {408, 425, 429}


def _intake_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/apis/intake/v2"):
        base = f"{base}/apis/intake/v2"
    return base


def _workspace_url(base_url: str, workspace: str, suffix: str) -> str:
    workspace_path = urllib.parse.quote(workspace, safe="")
    return f"{_intake_base(base_url)}/workspaces/{workspace_path}{suffix}"


def atif_ingest_url(base_url: str, workspace: str) -> str:
    return _workspace_url(base_url, workspace, "/ingest/atif")


def create_experiment_group(
    base_url: str,
    workspace: str,
    name: str,
    metadata: dict[str, str],
    timeout: float,
) -> str:
    """Ensure the parent Experiment (one per benchmark) exists; return its entity id.

    Idempotent: a 409 means it already exists, so we GET it to read the id.
    """
    body: dict[str, Any] = {"name": name}
    if metadata:
        body["metadata"] = metadata
    status, response = request_json("POST", _workspace_url(base_url, workspace, "/experiments"), body, timeout)
    if status == 201 and response is not None:
        return str(response["id"])
    if status == 409:
        name_path = urllib.parse.quote(name, safe="")
        status, response = request_json(
            "GET",
            _workspace_url(base_url, workspace, f"/experiments/{name_path}"),
            None,
            timeout,
        )
        if status == 200 and response is not None:
            return str(response["id"])
    raise IntakeError(
        f"experiment ensure failed for {name!r}: HTTP {status}; "
        f"attempts={_attempt_count_for_status(status)}: {response}"
    )


def create_evaluation(base_url: str, workspace: str, body: dict[str, Any], timeout: float) -> None:
    """Create an Evaluation. Idempotent: a 409 means a sibling upload already created it."""
    status, response = request_json("POST", _workspace_url(base_url, workspace, "/evaluations"), body, timeout)
    if status in {201, 409}:
        return
    raise IntakeError(
        f"evaluation create failed for {body.get('name')!r}: HTTP {status}; "
        f"attempts={_attempt_count_for_status(status)}: {response}"
    )


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> tuple[int, dict[str, Any] | None]:
    """Issue one Intake request with bounded retries for transient failures.

    Intake derives stable span and evaluator-result identities for identical
    ATIF payloads, so retrying an ambiguous ingest response does not duplicate
    the resulting trace data.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be non-negative")

    data = None
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                try:
                    return response.status, json.loads(body) if body else None
                except json.JSONDecodeError as exc:
                    raise IntakeError(f"Intake {method} returned invalid JSON: {body[:500]!r}") from exc
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                parsed: dict[str, Any] | None = json.loads(body) if body else None
            except json.JSONDecodeError:
                parsed = {"detail": body}
            if not _retryable_status(exc.code) or attempt == max_attempts:
                return exc.code, parsed
            _log_and_wait(method, url, attempt, max_attempts, retry_delay_seconds, exc)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == max_attempts:
                raise IntakeError(f"Intake {method} transport failed after {max_attempts} attempts: {exc}") from exc
            _log_and_wait(method, url, attempt, max_attempts, retry_delay_seconds, exc)

    raise AssertionError("unreachable Intake retry state")


def _retryable_status(status: int) -> bool:
    return status in RETRYABLE_HTTP_STATUSES or 500 <= status < 600


def _attempt_count_for_status(status: int) -> int:
    return DEFAULT_MAX_ATTEMPTS if _retryable_status(status) else 1


def _log_and_wait(
    method: str,
    url: str,
    attempt: int,
    max_attempts: int,
    retry_delay_seconds: float,
    error: Exception,
) -> None:
    delay = retry_delay_seconds * (2 ** (attempt - 1))
    LOG.warning(
        "Intake %s %s attempt %d/%d failed transiently: %s; retrying in %.1fs",
        method,
        url,
        attempt,
        max_attempts,
        error,
        delay,
    )
    time.sleep(delay)


def post_atif_payload(
    base_url: str,
    workspace: str,
    item: TrialPayload,
    timeout: float,
) -> None:
    status, body = request_json(
        "POST",
        atif_ingest_url(base_url, workspace),
        item.payload,
        timeout,
    )
    if status in {200, 201, 202, 204}:
        return
    raise IntakeError(
        f"ATIF ingest POST failed for {item.external_id}: HTTP {status}; "
        f"attempts={_attempt_count_for_status(status)}: {body}"
    )
