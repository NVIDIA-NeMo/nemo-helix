# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side checks for the in-container auth report. No platform required."""

from __future__ import annotations

from typing import Any

from probe import evaluate_auth_report, parse_auth_report, workload_probe_script


def _report(**overrides: object) -> dict[str, Any]:
    report: dict[str, Any] = {
        "kind": "job",
        "workspace": "check-1",
        "mode": "on_behalf_of",
        "principal": {"id": "user-1", "email": "user@example.com", "on_behalf_of": ""},
        "workload_token_file_exists": False,
        "workload_identity_requested": False,
        "workspace_get": {"ok": True, "name": "check-1"},
        "secret_expected": True,
        "secret_matches": True,
        "model_call": {"skipped": True},
    }
    report.update(overrides)
    return report


def test_probe_source_compiles() -> None:
    compile(workload_probe_script(), "<instance-check-probe>", "exec")


def test_parse_auth_report_uses_the_last_line() -> None:
    text = "\n".join(
        [
            "boot",
            'INSTANCE_CHECK_AUTH {"mode": "anonymous"}',
            'prefix INSTANCE_CHECK_AUTH {"mode": "on_behalf_of", "workspace_get": {"ok": true}}',
        ]
    )
    parsed = parse_auth_report(text)
    assert parsed is not None
    assert parsed["mode"] == "on_behalf_of"


def test_parse_auth_report_missing() -> None:
    assert parse_auth_report("no report here") is None


def test_anonymous_fails_only_when_auth_is_enabled() -> None:
    report = _report(mode="anonymous", principal=None)
    open_platform = evaluate_auth_report(
        report,
        auth_enabled=False,
        token_exchange_enabled=False,
        require_token_exchange=False,
        model_required=False,
    )
    assert open_platform.problems == []
    locked = evaluate_auth_report(
        report,
        auth_enabled=True,
        token_exchange_enabled=False,
        require_token_exchange=False,
        model_required=False,
    )
    assert any("anonymously" in problem for problem in locked.problems)


def test_auth_proxy_workspace_get_is_authenticated() -> None:
    report = _report(mode="anonymous", principal=None, auth_proxy=True)
    evaluation = evaluate_auth_report(
        report,
        auth_enabled=True,
        token_exchange_enabled=False,
        require_token_exchange=False,
        model_required=False,
    )
    assert evaluation.problems == []
    assert any("auth-proxy" in note for note in evaluation.notes)


def test_trusted_headers_accept_principal_injection() -> None:
    evaluation = evaluate_auth_report(
        _report(),
        auth_enabled=True,
        token_exchange_enabled=False,
        require_token_exchange=False,
        model_required=False,
    )
    assert evaluation.problems == []


def test_token_exchange_requires_a_workload_identity_token() -> None:
    report = _report()
    evaluation = evaluate_auth_report(
        report,
        auth_enabled=True,
        token_exchange_enabled=True,
        require_token_exchange=False,
        model_required=False,
    )
    assert any("workload-identity token" in problem for problem in evaluation.problems)
    assert evaluation.notes == []


def test_requested_workload_identity_requires_a_token_file() -> None:
    report = _report(workload_identity_requested=True)
    evaluation = evaluate_auth_report(
        report,
        auth_enabled=True,
        token_exchange_enabled=True,
        require_token_exchange=False,
        model_required=False,
    )
    assert any("workload identity" in problem for problem in evaluation.problems)


def test_workload_identity_mode_passes_when_the_token_file_exists() -> None:
    report = _report(mode="workload_identity", workload_token_file_exists=True, workload_identity_requested=True)
    evaluation = evaluate_auth_report(
        report,
        auth_enabled=True,
        token_exchange_enabled=True,
        require_token_exchange=True,
        model_required=False,
    )
    assert evaluation.problems == []


def test_secret_mismatch_and_failed_workspace_get_are_problems() -> None:
    report = _report(workspace_get={"ok": False, "error": "401"}, secret_matches=False)
    evaluation = evaluate_auth_report(
        report,
        auth_enabled=True,
        token_exchange_enabled=False,
        require_token_exchange=False,
        model_required=False,
    )
    assert any("workspace get" in problem for problem in evaluation.problems)
    assert any("secret" in problem for problem in evaluation.problems)


def test_model_call_is_required_only_when_asked() -> None:
    report = _report(model_call={"ok": False, "error": "404"})
    optional = evaluate_auth_report(
        report,
        auth_enabled=False,
        token_exchange_enabled=False,
        require_token_exchange=False,
        model_required=False,
    )
    required = evaluate_auth_report(
        report,
        auth_enabled=False,
        token_exchange_enabled=False,
        require_token_exchange=False,
        model_required=True,
    )
    assert optional.problems == []
    assert any("model call" in problem for problem in required.problems)
