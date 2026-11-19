# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time_machine
from nemo_observability_plugin.session import SessionSigner


def test_round_trip():
    signer = SessionSigner(b"k" * 32)
    token, session = signer.mint("user-1", "default", "mlflow", ttl_seconds=60)
    assert signer.verify(token, "mlflow") == session
    assert session.principal == "user-1"
    assert session.workspace == "default"


def test_wrong_connection_rejected():
    signer = SessionSigner()
    token, _ = signer.mint("user-1", "default", "mlflow", ttl_seconds=60)
    assert signer.verify(token, "phoenix") is None


def test_other_key_rejected():
    token, _ = SessionSigner(b"a" * 32).mint("user-1", "default", "mlflow", ttl_seconds=60)
    assert SessionSigner(b"b" * 32).verify(token, "mlflow") is None


def test_tampered_payload_rejected():
    signer = SessionSigner()
    user_token, _ = signer.mint("user-1", "default", "mlflow", ttl_seconds=60)
    admin_token, _ = signer.mint("admin", "default", "mlflow", ttl_seconds=60)
    forged = admin_token.split(".")[0] + "." + user_token.split(".")[1]
    assert signer.verify(forged, "mlflow") is None


def test_expired_rejected():
    signer = SessionSigner()
    with time_machine.travel(0, tick=False):
        token, _ = signer.mint("user-1", "default", "mlflow", ttl_seconds=60)
    with time_machine.travel(61, tick=False):
        assert signer.verify(token, "mlflow") is None
    with time_machine.travel(59, tick=False):
        assert signer.verify(token, "mlflow") is not None


def test_garbage_rejected():
    signer = SessionSigner()
    for token in (None, "", "abc", "a.b.c", "!!!.???"):
        assert signer.verify(token, "mlflow") is None
