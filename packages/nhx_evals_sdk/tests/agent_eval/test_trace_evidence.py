# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trace evidence the Fabric and Gym runtimes share."""

from __future__ import annotations

from pathlib import Path

from nhx_evals_sdk.agent_eval.runtimes.trace_evidence import ArtifactView, declared_artifacts, register_trace_evidence
from nhx_evals_sdk.values.evidence import EVIDENCE_FORMAT_ATIF, EVIDENCE_FORMAT_OTLP, EVIDENCE_TRACE


def test_the_primary_trace_is_otlp_when_there_is_one_and_atif_otherwise(tmp_path: Path) -> None:
    # ATIF needs no file on disk: Fabric declares it in the artifact manifest and the handle
    # validates it on read.
    atif, otlp = tmp_path / "t.atif.json", tmp_path / "relay.jsonl"

    atif_only: dict = {}
    register_trace_evidence(atif_only, atif=atif, otlp=otlp)
    assert atif_only[EVIDENCE_TRACE].format == EVIDENCE_FORMAT_ATIF

    otlp.write_text("{}\n", encoding="utf-8")
    both: dict = {}
    register_trace_evidence(both, atif=atif, otlp=otlp)
    assert both[EVIDENCE_TRACE].format == EVIDENCE_FORMAT_OTLP
    # The format decides which view is primary, never which is reachable.
    assert both[f"{EVIDENCE_TRACE}:{EVIDENCE_FORMAT_ATIF}"].format == EVIDENCE_FORMAT_ATIF
    assert both[f"{EVIDENCE_TRACE}:{EVIDENCE_FORMAT_OTLP}"].format == EVIDENCE_FORMAT_OTLP


def test_an_empty_trace_file_is_not_registered(tmp_path: Path) -> None:
    otlp = tmp_path / "relay.jsonl"
    otlp.touch()

    descriptors: dict = {}
    register_trace_evidence(descriptors, atif=None, otlp=otlp)

    assert descriptors == {}


def test_declared_artifacts_skips_entries_without_a_name_or_path() -> None:
    result = {
        "artifacts": {
            "artifacts": [
                {"name": "relay", "path": "/run/relay.jsonl", "kind": "otlp", "media_type": "application/jsonl"},
                {"name": "no-path"},
                {"path": "/run/no-name"},
                "not-an-entry",
                {"name": "bare", "path": "/run/bare"},
            ]
        }
    }

    assert declared_artifacts(result) == (
        ArtifactView(name="relay", kind="otlp", path="/run/relay.jsonl", media_type="application/jsonl"),
        ArtifactView(name="bare", kind=None, path="/run/bare", media_type=None),
    )


def test_declared_artifacts_drops_what_the_path_mapper_did_not_bring_across() -> None:
    result = {
        "artifacts": {
            "artifacts": [
                {"name": "kept", "path": "/sandbox/kept"},
                {"name": "lost", "path": "/sandbox/lost"},
            ]
        }
    }

    def to_host(path: str) -> str | None:
        return path.replace("/sandbox", "/host") if path.endswith("kept") else None

    assert [(a.name, a.path) for a in declared_artifacts(result, artifact_path=to_host)] == [("kept", "/host/kept")]


def test_declared_artifacts_is_empty_without_a_manifest() -> None:
    assert declared_artifacts({}) == ()
    assert declared_artifacts({"artifacts": "nonsense"}) == ()
