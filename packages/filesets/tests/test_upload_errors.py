# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Upload errors survive the trip across a process boundary."""

from __future__ import annotations

import pickle

from filesets import UploadRetriesExhaustedError


def test_upload_retries_exhausted_error_pickles() -> None:
    error = pickle.loads(pickle.dumps(UploadRetriesExhaustedError("local.bin", "default/set/remote.bin", 3)))

    assert (error.lpath, error.rpath, error.attempts) == ("local.bin", "default/set/remote.bin", 3)
    assert str(error) == "Upload of local.bin to default/set/remote.bin failed after 3 attempts"


def test_upload_retries_exhausted_error_keeps_last_failure_across_pickle() -> None:
    try:
        try:
            raise RuntimeError("HTTP 503: storage throttled")
        except RuntimeError as exc:
            raise UploadRetriesExhaustedError("local.bin", "default/set/remote.bin", 3, str(exc)) from exc
    except UploadRetriesExhaustedError as raised:
        error = pickle.loads(pickle.dumps(raised))

    assert error.__cause__ is None
    assert error.last_error == "HTTP 503: storage throttled"
    assert str(error) == (
        "Upload of local.bin to default/set/remote.bin failed after 3 attempts: HTTP 503: storage throttled"
    )
