# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the convention frames in :mod:`nemo_helix_plugin.functions.frames`.

The frames are tiny Pydantic models — these tests document the wire
shape (the ``kind`` discriminator + each frame's required fields) so a
silent rename or default change shows up here first.
"""

from __future__ import annotations

import json

from nemo_helix_plugin.functions.frames import Done, Error, Heartbeat


def test_heartbeat_default_kind() -> None:
    assert Heartbeat().model_dump() == {"kind": "heartbeat"}


def test_done_default_kind() -> None:
    assert Done().model_dump() == {"kind": "done"}


def test_error_carries_message_and_optional_details() -> None:
    err = Error(message="boom")
    assert err.kind == "error"
    assert err.message == "boom"
    assert err.details is None

    err_with_details = Error(message="boom", details={"reason": "out of memory"})
    assert err_with_details.details == {"reason": "out of memory"}


def test_frames_round_trip_through_ndjson() -> None:
    """Producer-side: ``model_dump_json``. Consumer side: ``model_validate_json``."""
    frames = [Heartbeat(), Heartbeat(), Done()]
    encoded = "\n".join(f.model_dump_json() for f in frames) + "\n"
    decoded = [json.loads(ln) for ln in encoded.splitlines() if ln]
    assert decoded == [{"kind": "heartbeat"}, {"kind": "heartbeat"}, {"kind": "done"}]


def test_kind_is_required_in_the_schema_despite_its_default() -> None:
    """An optional discriminator does not discriminate — clients could not narrow."""
    for frame in (Heartbeat, Done, Error):
        assert "kind" in frame.model_json_schema()["required"]


def test_nullable_fields_stay_optional() -> None:
    """Promoting every defaulted field would make these required but non-nullable,
    because the spec pipeline drops the ``null`` branch."""
    assert "details" not in Error.model_json_schema()["required"]
