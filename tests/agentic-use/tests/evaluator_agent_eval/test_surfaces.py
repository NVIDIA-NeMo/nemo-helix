# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator_agent_eval.surfaces."""

from evaluator_agent_eval.surfaces import SurfaceEvidence, detect_surfaces


def test_surface_detection_uses_logs_commands_and_forbidden_patterns():
    result = detect_surfaces(
        SurfaceEvidence(
            final_answer_text="Use packages/nhx_evals_sdk.",
            raw_logs=["ran nemo evaluation metrics run"],
            command_argvs=[["nemo", "evaluation", "metrics", "run"]],
            changed_paths=["services/evaluator/old.py"],
        ),
        forbidden_patterns=["services/", "nemo evaluation"],
    )

    assert result.observed_surfaces == ["standalone_sdk", "cli", "legacy_service"]
    assert result.forbidden_surface_hits == ["services/", "nemo evaluation"]


def test_surface_detection_defaults_to_unknown():
    result = detect_surfaces(SurfaceEvidence())

    assert result.observed_surfaces == ["unknown"]
    assert result.forbidden_surface_hits == []


def test_surface_detection_does_not_treat_json_escaped_newline_as_service_path():
    result = detect_surfaces(
        SurfaceEvidence(
            final_answer_text="This uses local SDK APIs, no remote services\\n2. ExactMatchMetric does the comparison.",
        )
    )

    assert result.observed_surfaces == ["unknown"]


def test_surface_detection_detects_windows_service_path():
    result = detect_surfaces(SurfaceEvidence(final_answer_text="I inspected services\\evaluator\\old.py."))

    assert result.observed_surfaces == ["legacy_service"]


def test_surface_detection_ignores_negated_forbidden_surface_mentions():
    result = detect_surfaces(
        SurfaceEvidence(
            final_answer_text=(
                "Use only packages/nhx_evals_sdk. Do not use the nemo evaluation CLI, plugin SDK APIs, or services/*."
            )
        ),
        forbidden_patterns=["nemo evaluation", "plugin sdk", "services/*"],
    )

    assert result.observed_surfaces == ["standalone_sdk"]
    assert result.forbidden_surface_hits == []
