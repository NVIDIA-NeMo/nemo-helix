# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``nemo_agents_plugin.usage.parser``."""

from __future__ import annotations

from pathlib import Path

import pytest
from nemo_agents_plugin.usage.models import BatchUsageReport, UsageReport
from nemo_agents_plugin.usage.parser import UsageParseError, parse_path


def test_parse_path_with_result_json_file(fixtures_dir: Path) -> None:
    """A direct path to result.json yields a single UsageReport."""
    report = parse_path(fixtures_dir / "result-ok-with-tokens.json")

    assert isinstance(report, UsageReport)
    assert report.task.task == "workspace-basic-mcp"
    assert report.task.prompt_tokens == 1500
    assert report.task.completion_tokens == 500
    assert report.task.total_tokens == 2000
    assert report.task.reward == 1
    assert report.task.agent_status == "ok"
    assert report.task.compute_units is None  # parser doesn't score


def test_parse_path_with_run_dir(tmp_run_dir: Path) -> None:
    """A directory containing result.json yields a single UsageReport."""
    report = parse_path(tmp_run_dir)

    assert isinstance(report, UsageReport)
    assert report.task.task == "workspace-basic-mcp"
    assert Path(report.task.source_dir) == tmp_run_dir.resolve()


def test_parse_path_with_natjobs_dir(tmp_natjobs_dir: Path) -> None:
    """A nat-jobs/ tree yields a BatchUsageReport with all runs."""
    report = parse_path(tmp_natjobs_dir)

    assert isinstance(report, BatchUsageReport)
    assert len(report.runs) == 4
    # Sorted by timestamp ascending — first is the 22:00:00Z fixture
    assert report.runs[0].timestamp == "20260429T220000Z"
    assert report.runs[-1].timestamp == "20260429T231000Z"

    # Token totals are None when any run has missing usage; null_token_runs
    # surfaces the count.  ``compute_units_total`` follows the same gate.
    assert report.prompt_tokens_total is None
    assert report.completion_tokens_total is None
    assert report.total_tokens_total is None
    assert report.null_token_runs == 3
    assert report.compute_units_total is None


def test_parse_path_with_natjobs_dir_all_runs_tokened(tmp_path: Path, fixtures_dir: Path) -> None:
    """When every run has tokens, totals are populated as a real sum."""
    natjobs = tmp_path / "nat-jobs"
    natjobs.mkdir()
    src = (fixtures_dir / "result-ok-with-tokens.json").read_text()
    for i, ts in enumerate(("20260429T220000Z", "20260429T230000Z", "20260429T240000Z")):
        run = natjobs / f"{ts}-task-{i}"
        run.mkdir()
        # Each run pulls the tokened fixture; totals sum cleanly
        (run / "result.json").write_text(src.replace("20260429T220000Z", ts))

    report = parse_path(natjobs)
    assert isinstance(report, BatchUsageReport)
    assert report.prompt_tokens_total == 4500  # 3 * 1500
    assert report.completion_tokens_total == 1500
    assert report.total_tokens_total == 6000
    assert report.null_token_runs == 0


def test_parse_path_preserves_null_tokens(fixtures_dir: Path) -> None:
    """`metrics.*_tokens: null` is preserved (not coerced to 0)."""
    report = parse_path(fixtures_dir / "result-ok-null-tokens.json")

    assert isinstance(report, UsageReport)
    assert report.task.prompt_tokens is None
    assert report.task.completion_tokens is None
    assert report.task.total_tokens is None
    assert report.task.reward == 1


def test_parse_path_preserves_failure_states(fixtures_dir: Path) -> None:
    """A failed-agent fixture parses with build=ok, agent=failed, verify=null."""
    report = parse_path(fixtures_dir / "result-failed-agent.json")

    assert isinstance(report, UsageReport)
    assert report.task.build_status == "ok"
    assert report.task.agent_status == "failed"
    assert report.task.verify_status is None
    assert report.task.reward == 0


def test_parse_path_handles_error_build(fixtures_dir: Path) -> None:
    """An error-build fixture parses with build_status carrying the error string.

    Distinguishes "build failed" from "agent failed" — both have null
    agent/verify, but build_status separates the two failure modes.
    """
    report = parse_path(fixtures_dir / "result-error-build.json")

    assert isinstance(report, UsageReport)
    assert report.task.build_status == "error: docker build failed"
    assert report.task.agent_status is None
    assert report.task.verify_status is None
    assert report.task.reward is None
    assert report.task.image == "nhx-nat-models-list-mcp:latest"


def test_parse_path_skips_subdirs_without_result(tmp_path: Path, fixtures_dir: Path) -> None:
    """Children without result.json are counted in skipped_runs and null totals.

    A 1-of-2 batch returns the one valid run but nulls token totals — a
    partial sum would silently masquerade as a complete batch.
    """
    natjobs = tmp_path / "nat-jobs"
    natjobs.mkdir()

    run = natjobs / "20260429T220000Z-task-a"
    run.mkdir()
    (run / "result.json").write_text((fixtures_dir / "result-ok-with-tokens.json").read_text())

    junk = natjobs / "20260429T230000Z-junk"
    junk.mkdir()
    (junk / "instruction.md").write_text("not a result file")

    report = parse_path(natjobs)
    assert isinstance(report, BatchUsageReport)
    assert len(report.runs) == 1
    assert report.skipped_runs == 1
    assert report.prompt_tokens_total is None
    assert report.completion_tokens_total is None
    assert report.total_tokens_total is None


def test_parse_path_rejects_empty_directory(tmp_path: Path) -> None:
    """An empty directory raises UsageParseError."""
    with pytest.raises(UsageParseError):
        parse_path(tmp_path)


def test_parse_path_rejects_non_existent(tmp_path: Path) -> None:
    """A non-existent path raises UsageParseError."""
    with pytest.raises(UsageParseError):
        parse_path(tmp_path / "does-not-exist")


def test_parse_path_rejects_invalid_json(tmp_path: Path) -> None:
    """A result.json with malformed content raises UsageParseError."""
    bad = tmp_path / "result.json"
    bad.write_text("{not valid json")

    with pytest.raises(UsageParseError):
        parse_path(bad)


def test_parse_path_rejects_non_dict_json(tmp_path: Path) -> None:
    """JSON that decodes to a non-object (list, scalar, null) raises UsageParseError."""
    for body in ('["a", "list"]', '"a string"', "42", "null"):
        bad = tmp_path / "result.json"
        bad.write_text(body)
        with pytest.raises(UsageParseError, match="expected JSON object"):
            parse_path(bad)


def test_parse_path_rejects_unrelated_json_dict(tmp_path: Path) -> None:
    """An unrelated JSON config without recognized keys is rejected (not silently coerced)."""
    bad = tmp_path / "result.json"
    bad.write_text('{"some_other_key": "value", "version": 1}')

    with pytest.raises(UsageParseError, match="not a result.json"):
        parse_path(bad)


def test_parse_path_rejects_partial_recognized_keys(tmp_path: Path) -> None:
    """A JSON object with only ``task`` (no ``metrics``) is rejected.

    Singletons like ``{"task": "..."}`` appear in unrelated configs (k8s
    manifests, CI matrices, agent configs).  The recognition gate requires
    both anchor keys to avoid false positives.
    """
    for body in ('{"task": "t"}', '{"metrics": {}}', '{"agent": "ok", "build": "ok"}'):
        bad = tmp_path / "result.json"
        bad.write_text(body)
        with pytest.raises(UsageParseError, match="missing required key"):
            parse_path(bad)


def test_parse_path_rejects_metrics_non_dict(tmp_path: Path) -> None:
    """A result.json whose ``metrics`` is a non-object (list, scalar) raises UsageParseError.

    Includes the falsy cases (``[]``, ``""``, ``0``) — a permissive ``or {}``
    coercion would silently flatten these to no-tokens; the explicit type
    check rejects them instead.
    """
    for body in (
        '{"task": "t", "metrics": [1, 2, 3]}',
        '{"task": "t", "metrics": []}',
        '{"task": "t", "metrics": ""}',
        '{"task": "t", "metrics": 0}',
    ):
        bad = tmp_path / "result.json"
        bad.write_text(body)
        with pytest.raises(UsageParseError, match="'metrics' must be a JSON object"):
            parse_path(bad)


def test_parse_path_rejects_pydantic_validation_failure(tmp_path: Path) -> None:
    """Pydantic ValidationError (non-coercible token field) is wrapped as UsageParseError."""
    bad = tmp_path / "result.json"
    # prompt_tokens=[1,2,3] cannot coerce to int|None
    bad.write_text('{"task": "t", "metrics": {"prompt_tokens": [1, 2, 3]}}')

    with pytest.raises(UsageParseError, match="schema mismatch"):
        parse_path(bad)


def test_parse_path_batch_skips_malformed_result_jsons(tmp_path: Path, fixtures_dir: Path) -> None:
    """A truncated/malformed result.json in one child does NOT abort the whole batch.

    Sibling runs still produce a useful report; the count surfaces via
    ``unparseable_runs``.
    """
    natjobs = tmp_path / "nat-jobs"
    natjobs.mkdir()

    good = natjobs / "20260429T220000Z-good"
    good.mkdir()
    (good / "result.json").write_text((fixtures_dir / "result-ok-with-tokens.json").read_text())

    truncated = natjobs / "20260429T230000Z-truncated"
    truncated.mkdir()
    (truncated / "result.json").write_text('{"task": "x", "metrics":')

    no_result = natjobs / "20260429T240000Z-no-result"
    no_result.mkdir()  # missing result.json — separate skip path from the truncated case

    report = parse_path(natjobs)
    assert isinstance(report, BatchUsageReport)
    assert len(report.runs) == 1
    assert report.runs[0].task == "workspace-basic-mcp"
    assert report.unparseable_runs == 1
    assert report.skipped_runs == 1
    # Both skipped and unparseable runs null the totals.
    assert report.prompt_tokens_total is None
    assert report.total_tokens_total is None


def test_parse_path_batch_skips_unreadable_result_jsons(tmp_path: Path, fixtures_dir: Path) -> None:
    """A chmod-000 ``result.json`` in one child does NOT abort the whole batch.

    OS-level read failures (``PermissionError`` / ``OSError`` / non-UTF-8 bytes
    via ``UnicodeDecodeError``) surface as ``UsageParseError`` so the batch
    loop counts them in ``unparseable_runs`` instead of letting one bad child
    tear down N-1 valid sibling runs.
    """
    natjobs = tmp_path / "nat-jobs"
    natjobs.mkdir()

    good = natjobs / "20260429T220000Z-good"
    good.mkdir()
    (good / "result.json").write_text((fixtures_dir / "result-ok-with-tokens.json").read_text())

    unreadable = natjobs / "20260429T230000Z-unreadable"
    unreadable.mkdir()
    bad_result = unreadable / "result.json"
    bad_result.write_text("{}")
    bad_result.chmod(0o000)
    try:
        report = parse_path(natjobs)
    finally:
        # Restore perms so pytest's tmp_path cleanup can remove the file.
        bad_result.chmod(0o644)

    assert isinstance(report, BatchUsageReport)
    assert len(report.runs) == 1
    assert report.runs[0].task == "workspace-basic-mcp"
    assert report.unparseable_runs == 1
    assert report.skipped_runs == 0


def test_parse_path_batch_all_unparseable_raises(tmp_path: Path) -> None:
    """If every child's result.json fails to parse, raise — there's nothing to report."""
    natjobs = tmp_path / "nat-jobs"
    natjobs.mkdir()
    bad = natjobs / "20260429T220000Z-bad"
    bad.mkdir()
    (bad / "result.json").write_text("{not valid")

    with pytest.raises(UsageParseError, match="no parseable result.json"):
        parse_path(natjobs)


def test_parse_path_warns_when_top_result_shadows_run_subdirs(tmp_path: Path, fixtures_dir: Path, caplog) -> None:
    """Directory containing both a top-level result.json AND run-shaped subdirs warns."""
    parent = tmp_path / "snapshot"
    parent.mkdir()
    (parent / "result.json").write_text((fixtures_dir / "result-ok-with-tokens.json").read_text())
    sibling = parent / "20260429T230000Z-other"
    sibling.mkdir()
    (sibling / "result.json").write_text((fixtures_dir / "result-ok-with-tokens.json").read_text())

    import logging

    with caplog.at_level(logging.WARNING):
        report = parse_path(parent)

    assert isinstance(report, UsageReport)
    assert any("top-level result.json" in rec.message for rec in caplog.records)


def test_parse_path_batch_partial_null_token_fields(tmp_path: Path, fixtures_dir: Path) -> None:
    """A run with only some token fields null counts toward null_token_runs.

    Guards against the failure mode where ``null_token_runs`` keys only on
    ``total_tokens`` while ``prompt_tokens_total`` (et al.) gate on their
    own field independently — a downstream consumer would see
    ``null_token_runs == 0`` and conclude the totals were complete sums.
    """
    natjobs = tmp_path / "nat-jobs"
    natjobs.mkdir()
    full_run = natjobs / "20260429T220000Z-full"
    full_run.mkdir()
    (full_run / "result.json").write_text((fixtures_dir / "result-ok-with-tokens.json").read_text())
    partial_run = natjobs / "20260429T230000Z-partial"
    partial_run.mkdir()
    # Only prompt_tokens is null; completion + total are populated.
    (partial_run / "result.json").write_text(
        '{"task": "partial", "timestamp": "20260429T230000Z", '
        '"metrics": {"prompt_tokens": null, "completion_tokens": 500, "total_tokens": 2500}}'
    )

    report = parse_path(natjobs)
    assert isinstance(report, BatchUsageReport)
    assert report.null_token_runs == 1  # the partial run is counted
    assert report.prompt_tokens_total is None
    assert report.completion_tokens_total is None
    assert report.total_tokens_total is None
