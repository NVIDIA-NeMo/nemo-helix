# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import json
import os
import time
from pathlib import Path

import pytest
from nemo_evaluator_sdk.execution.samples import build_metric_input
from nemo_evaluator_sdk.values.evidence import (
    EVIDENCE_FORMAT_ATIF,
    EVIDENCE_FORMAT_OTLP,
    ATIFTraceHandle,
    CandidateEvidence,
    EvidenceDescriptor,
    LocalFilesystemEvidence,
    OTLPTraceHandle,
)
from pydantic import JsonValue, ValidationError


def test_metric_input_preserves_candidate_evidence_out_of_metadata() -> None:
    evidence = CandidateEvidence(
        descriptors={"trace": EvidenceDescriptor(kind="atif", ref="atif://trial-trace#L9", format="atif")}
    )

    metric_input = build_metric_input(
        {"prompt": "Question?"},
        {"output_text": "Answer", "evidence": evidence, "custom": "metadata"},
        index=3,
    )

    assert metric_input.candidate.evidence == evidence
    assert metric_input.candidate.evidence is not None
    assert metric_input.candidate.evidence.require("trace", kind="atif") == evidence.descriptors["trace"]
    assert metric_input.candidate.metadata == {"custom": "metadata"}


@pytest.mark.asyncio
async def test_candidate_evidence_filesystem_access_is_lazy_and_cached(tmp_path: Path) -> None:
    final_state = tmp_path / "final"
    final_state.mkdir()
    (final_state / "answer.txt").write_text("done", encoding="utf-8")
    (final_state / "nested").mkdir()
    (final_state / "nested" / "notes.txt").write_text("notes", encoding="utf-8")

    evidence = CandidateEvidence(
        descriptors={
            "remote_state": EvidenceDescriptor(kind="filesystem", ref="https://example.test/archive.tgz"),
            "final_state": EvidenceDescriptor(kind="filesystem", ref=str(final_state)),
        }
    )

    assert evidence.require("remote_state", kind="filesystem").ref == "https://example.test/archive.tgz"

    handle = await evidence.filesystem("final_state")
    cached = await evidence.filesystem("final_state")

    assert handle is cached
    assert await handle.exists("answer.txt") is True
    assert await handle.read_text("answer.txt") == "done"
    assert await handle.iter_paths(recursive=True) == ["answer.txt", "nested", "nested/notes.txt"]
    with pytest.raises(ValueError, match="outside evidence root"):
        handle.path("../outside")
    with pytest.raises(ValueError, match="only supports local filesystem"):
        await evidence.filesystem("remote_state")


@pytest.mark.asyncio
async def test_filesystem_read_bytes_list_and_diff(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    for root in (before, after):
        (root / "src").mkdir(parents=True)
    (before / "keep.txt").write_text("same", encoding="utf-8")
    (before / "src" / "mod.py").write_text("old", encoding="utf-8")
    (before / "gone.txt").write_text("bye", encoding="utf-8")
    (after / "keep.txt").write_text("same", encoding="utf-8")
    (after / "src" / "mod.py").write_text("new", encoding="utf-8")
    (after / "added.txt").write_text("hi", encoding="utf-8")

    initial = LocalFilesystemEvidence(before)
    final = LocalFilesystemEvidence(after)

    assert await final.read_bytes("added.txt") == b"hi"
    assert await final.list_files("**/*.py") == ["src/mod.py"]

    diff = await initial.diff(final)
    assert {(entry.path, entry.change_type) for entry in diff.entries} == {
        ("added.txt", "added"),
        ("gone.txt", "deleted"),
        ("src/mod.py", "modified"),
    }
    assert [entry.path for entry in diff.changed(prefix="src/", kinds={"modified"})] == ["src/mod.py"]


@pytest.mark.asyncio
async def test_run_verifier_uses_overlay_and_reports_outcome(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "answer.txt").write_text("42", encoding="utf-8")
    handle = LocalFilesystemEvidence(root)

    ok = await handle.run_verifier(["cat", "answer.txt"])
    assert ok.ok and ok.exit_code == 0 and ok.stdout.strip() == "42"

    failed = await handle.run_verifier(["false"])
    assert not failed.ok and failed.exit_code != 0

    timed_out = await handle.run_verifier(["sleep", "5"], timeout_s=0.2)
    assert timed_out.timed_out and not timed_out.ok

    # The verifier ran in a throwaway copy, so the stored evidence is untouched.
    await handle.run_verifier(["sh", "-c", "echo cheat > sneaked.txt"])
    assert await handle.list_files("**/*") == ["answer.txt"]


@pytest.mark.asyncio
async def test_run_verifier_overlay_files_are_staged_over_the_copy(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "answer.txt").write_text("agent", encoding="utf-8")  # the agent could edit this
    handle = LocalFilesystemEvidence(root)

    # A held-out file the agent never had: it appears in the copy the verifier runs against.
    made = await handle.run_verifier(["cat", "held_out.txt"], overlay_files={"held_out.txt": "truth"})
    assert made.ok and made.stdout.strip() == "truth"

    # An overlay file wins over the agent's version (so held-out ground truth can't be edited away).
    over = await handle.run_verifier(["cat", "answer.txt"], overlay_files={"answer.txt": "truth"})
    assert over.stdout.strip() == "truth"

    # Overlaying does not mutate the stored evidence, and it accepts nested paths.
    nested = await handle.run_verifier(["cat", "tests/t.txt"], overlay_files={"tests/t.txt": "ok"})
    assert nested.stdout.strip() == "ok"
    assert await handle.list_files("**/*") == ["answer.txt"]


@pytest.mark.asyncio
async def test_run_verifier_overlay_rejects_path_escaping_the_copy(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    handle = LocalFilesystemEvidence(root)

    with pytest.raises(ValueError, match="escapes the verifier copy"):
        await handle.run_verifier(["true"], overlay_files={"../escape.txt": "nope"})


@pytest.mark.asyncio
async def test_escaping_symlinks_are_not_hashed_or_copied(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET", encoding="utf-8")

    root = tmp_path / "workspace"
    root.mkdir()
    (root / "answer.txt").write_text("42", encoding="utf-8")
    (root / "leak").symlink_to(secret)  # agent-planted link escaping the evidence root
    handle = LocalFilesystemEvidence(root)

    # Listing and content hashing ignore the escaping symlink (no host-file read/leak).
    assert await handle.list_files("**/*") == ["answer.txt"]
    assert (await handle.diff(LocalFilesystemEvidence(root))).entries == []

    # The verifier overlay never receives the escaping link, so its target can't be read.
    leaked = await handle.run_verifier(["cat", "leak"])
    assert not leaked.ok


@pytest.mark.asyncio
async def test_absolute_symlinks_are_not_copied_into_verifier_overlay(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "answer.txt").write_text("42", encoding="utf-8")
    # Absolute link whose target is *inside* the root: passes _within_root, but copytree
    # symlinks=True would recreate it verbatim and a write through it would hit stored evidence.
    (root / "writable").symlink_to((root / "answer.txt").resolve())
    handle = LocalFilesystemEvidence(root)

    await handle.run_verifier(["sh", "-c", "echo pwned > writable"])

    # The write landed in the throwaway overlay, never on the real answer.txt.
    assert (root / "answer.txt").read_text(encoding="utf-8") == "42"


@pytest.mark.asyncio
async def test_verifier_timeout_kills_the_whole_process_tree(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    handle = LocalFilesystemEvidence(root)

    # The verifier backgrounds a long-lived child and records its PID. Killing only the
    # direct shell would orphan that child; killing the whole process group reaps it.
    # (We assert the child PID is gone rather than watching for a follow-on side effect,
    # which would race the kill: reaping the child can let the shell run its next command
    # in the window before the shell itself is signalled.)
    pidfile = tmp_path / "child.pid"
    result = await handle.run_verifier(["sh", "-c", f"sleep 30 & echo $! > '{pidfile}'; wait"], timeout_s=0.3)
    assert result.timed_out

    child_pid = int(pidfile.read_text().strip())
    # Poll until the killed group is fully reaped rather than assuming a fixed window;
    # on a busy runner the child can briefly linger as a zombie even when cleanup is correct.
    deadline = time.monotonic() + 2.0
    while True:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        if time.monotonic() >= deadline:
            pytest.fail(f"child process {child_pid} still alive after cleanup timeout")
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_unified_diff_reports_text_patch_and_skips_binary(tmp_path: Path) -> None:
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    for root in (before_root, after_root):
        root.mkdir()
    (before_root / "f.txt").write_text("a\nb\n", encoding="utf-8")
    (after_root / "f.txt").write_text("a\nc\n", encoding="utf-8")
    (before_root / "img.bin").write_bytes(b"\xff\xfe\x00")
    (after_root / "img.bin").write_bytes(b"\xff\xfe\x01")
    before, after = LocalFilesystemEvidence(before_root), LocalFilesystemEvidence(after_root)

    patch = await before.unified_diff(after, "f.txt")
    assert "-b" in patch and "+c" in patch and patch.startswith("--- a/f.txt")
    assert await before.unified_diff(after, "img.bin") == ""  # binary: no textual patch
    assert await before.unified_diff(before, "f.txt") == ""  # identical: empty


_ATIF_TRAJECTORY = {
    "schema_version": "ATIF-v1.7",
    "session_id": "session-1",
    "trajectory_id": "trajectory-1",
    "agent": {
        "name": "demo",
        "version": "1.0",
        "model_name": "demo-model",
        "tool_definitions": [{"type": "function", "function": {"name": "search"}}],
        "extra": {"deployment": "local"},
        "future_agent_field": "accepted",
    },
    "steps": [
        {
            "step_id": 1,
            "timestamp": "2026-06-30T12:00:00Z",
            "source": "user",
            "message": "do it",
            "extra": {"request_id": "request-1"},
            "future_step_field": "accepted",
        },
        {
            "step_id": 2,
            "timestamp": "2026-06-30T12:00:01Z",
            "source": "agent",
            "model_name": "demo-model",
            "reasoning_effort": "low",
            "message": "calling tool",
            "reasoning_content": "need current data",
            "tool_calls": [
                {
                    "tool_call_id": "c1",
                    "function_name": "search",
                    "arguments": {"q": "x"},
                    "extra": {"event_type": "function_complete"},
                    "future_tool_call_field": "accepted",
                }
            ],
            "observation": {
                "results": [
                    {
                        "source_call_id": "c1",
                        "content": '{"count": 4}',
                        "extra": {"event_type": "function_complete"},
                        "future_result_field": "accepted",
                    }
                ],
                "future_observation_field": "accepted",
            },
            "metrics": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cached_tokens": 3,
                "cost_usd": 0.01,
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3],
                "logprobs": [-0.5],
                "extra": {"provider": "test"},
            },
            "is_copied_context": False,
            "llm_call_count": 1,
            "extra": {"stream_updates": [{"sequence": 1}, {"sequence": 2}]},
        },
    ],
    "notes": "tool evidence fixture",
    "final_metrics": {
        "total_prompt_tokens": 10,
        "total_completion_tokens": 5,
        "total_cached_tokens": 3,
        "total_cost_usd": 0.01,
        "total_steps": 2,
        "extra": {"provider": "test"},
    },
    "continued_trajectory_ref": "trajectory-2.json",
    "extra": {"translator": "bugnemo"},
    "future_root_field": "accepted",
}


@pytest.mark.asyncio
async def test_trace_handle_reads_atif(tmp_path: Path) -> None:
    trace_path = tmp_path / "trajectory.json"
    trace_path.write_text(json.dumps(_ATIF_TRAJECTORY), encoding="utf-8")

    evidence = CandidateEvidence(
        descriptors={"trace": EvidenceDescriptor(kind="trace", ref=str(trace_path), format="atif")}
    )
    handle = await evidence.trace("trace")
    assert isinstance(handle, ATIFTraceHandle)
    assert handle.format == "atif"
    assert handle is await evidence.trace("trace")  # cached
    trajectory = await handle.trace()
    assert trajectory.schema_version == "ATIF-v1.7"
    assert [step.source for step in trajectory.steps] == ["user", "agent"]
    assert (await handle.tool_calls())[0].function_name == "search"
    assert (await handle.token_usage()).total_prompt_tokens == 10

    # A non-conformant trace is rejected at read time (producers must emit ATIF).
    bad = CandidateEvidence(
        descriptors={"trace": EvidenceDescriptor(kind="trace", format="atif", data={"steps": "not-a-list"})}
    )
    bad_handle = await bad.trace("trace")
    assert isinstance(bad_handle, ATIFTraceHandle)
    with pytest.raises(ValidationError):
        await bad_handle.trace()


@pytest.mark.asyncio
async def test_trace_handle_reads_atif_when_optional_format_is_absent() -> None:
    evidence = CandidateEvidence(descriptors={"trace": EvidenceDescriptor(kind="trace", data=_ATIF_TRAJECTORY)})

    handle = await evidence.trace("trace")
    assert isinstance(handle, ATIFTraceHandle)
    assert handle.format == "atif"
    trajectory = await handle.trace()

    assert trajectory.schema_version == "ATIF-v1.7"


@pytest.mark.asyncio
async def test_trace_handle_reads_otlp_jsonl_and_inline_data(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps({"resourceSpans": [{"schemaUrl": "keep-me", "scopeSpans": []}]})
        + "\n"
        + json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [{"name": "child"}]}]}]})
        + "\n",
        encoding="utf-8",
    )
    evidence = CandidateEvidence(
        descriptors={"trace": EvidenceDescriptor(kind="trace", format=EVIDENCE_FORMAT_OTLP, ref=str(trace_path))}
    )

    assert EVIDENCE_FORMAT_ATIF == "atif"
    assert EVIDENCE_FORMAT_OTLP == "otlp"
    handle = await evidence.trace("trace")
    assert isinstance(handle, OTLPTraceHandle)
    assert handle.format == "otlp"
    assert handle is await evidence.trace("trace")

    resource_spans = await handle.resource_spans()
    assert len(resource_spans) == 2
    assert resource_spans[0].schema_url == "keep-me"
    assert resource_spans[1].scope_spans[0].spans[0].name == "child"

    inline = CandidateEvidence(
        descriptors={"trace": EvidenceDescriptor(kind="trace", format="otlp", data=[{"scopeSpans": []}])}
    )
    inline_handle = await inline.trace("trace")
    assert isinstance(inline_handle, OTLPTraceHandle)
    assert [rs.scope_spans for rs in await inline_handle.resource_spans()] == [[]]


@pytest.mark.asyncio
async def test_trace_handle_reads_pretty_printed_otlp_object(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps({"resourceSpans": [{"scopeSpans": []}]}, indent=2),
        encoding="utf-8",
    )
    evidence = CandidateEvidence(
        descriptors={"trace": EvidenceDescriptor(kind="trace", format="otlp", ref=str(trace_path))}
    )

    handle = await evidence.trace()
    assert isinstance(handle, OTLPTraceHandle)
    assert [rs.scope_spans for rs in await handle.resource_spans()] == [[]]


@pytest.mark.asyncio
async def test_trace_handle_caches_empty_otlp_after_first_read(tmp_path: Path) -> None:
    trace_path = tmp_path / "empty.jsonl"
    trace_path.write_text('{"resourceSpans": []}\n', encoding="utf-8")
    evidence = CandidateEvidence(
        descriptors={"trace": EvidenceDescriptor(kind="trace", format="otlp", ref=str(trace_path))}
    )

    handle = await evidence.trace()
    assert isinstance(handle, OTLPTraceHandle)
    assert await handle.resource_spans() == []
    trace_path.unlink()
    assert await handle.resource_spans() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        "not-an-object",
        {},
        {"resourceSpans": "not-a-list"},
        {"resourceSpans": ["not-an-object"]},
    ],
)
async def test_trace_handle_rejects_malformed_otlp(payload: JsonValue) -> None:
    evidence = CandidateEvidence(descriptors={"trace": EvidenceDescriptor(kind="trace", format="otlp", data=payload)})

    handle = await evidence.trace()
    assert isinstance(handle, OTLPTraceHandle)
    with pytest.raises(ValueError):
        await handle.resource_spans()


@pytest.mark.asyncio
async def test_trace_handle_rejects_unknown_format() -> None:
    evidence = CandidateEvidence(
        descriptors={"trace": EvidenceDescriptor(kind="trace", format="json", data={"resourceSpans": []})}
    )

    with pytest.raises(ValueError, match="unknown trace evidence format 'json'"):
        await evidence.trace()


@pytest.mark.asyncio
async def test_trace_handle_exposes_typed_tool_evidence_and_retains_modeled_fields() -> None:
    evidence = CandidateEvidence(
        descriptors={"trace": EvidenceDescriptor(kind="trace", format="atif", data=_ATIF_TRAJECTORY)}
    )

    handle = await evidence.trace("trace")
    assert isinstance(handle, ATIFTraceHandle)
    trajectory = await handle.trace()
    assert trajectory.agent is not None
    assert trajectory.agent.name == "demo"
    assert trajectory.session_id == "session-1"
    assert trajectory.trajectory_id == "trajectory-1"
    assert trajectory.steps[1].step_id == 2
    assert trajectory.steps[1].observation is not None
    result = trajectory.steps[1].observation.results[0]
    assert result.source_call_id == "c1"
    assert result.content == '{"count": 4}'
    assert result.extra == {"event_type": "function_complete"}
    assert trajectory.steps[1].extra == {"stream_updates": [{"sequence": 1}, {"sequence": 2}]}
    assert trajectory.steps[1].tool_calls is not None
    assert trajectory.steps[1].tool_calls[0].extra == {"event_type": "function_complete"}
    assert trajectory.steps[1].metrics is not None
    assert trajectory.steps[1].metrics.cached_tokens == 3
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_steps == 2

    dumped = trajectory.model_dump(exclude_none=True)
    assert dumped["agent"]["extra"] == {"deployment": "local"}
    assert dumped["steps"][1]["observation"]["results"][0]["extra"] == {"event_type": "function_complete"}
    assert dumped["steps"][1]["extra"] == {"stream_updates": [{"sequence": 1}, {"sequence": 2}]}
    assert dumped["steps"][1]["tool_calls"][0]["extra"] == {"event_type": "function_complete"}
    assert dumped["extra"] == {"translator": "bugnemo"}

    # Unknown fields from newer ATIF revisions are accepted but omitted from the
    # typed read view; the producer dictionary remains authoritative elsewhere.
    assert "future_root_field" not in dumped
    assert "future_step_field" not in dumped["steps"][0]
    assert "future_result_field" not in dumped["steps"][1]["observation"]["results"][0]


def test_observation_models_are_exported_from_source_and_vendored_values_packages() -> None:
    from nemo_evaluator_sdk.values import Observation as SourceObservation
    from nemo_evaluator_sdk.values import ObservationResult as SourceObservationResult

    vendored_values = importlib.import_module("nemo_helix.beta.evaluator.values")
    VendoredObservation = vendored_values.Observation
    VendoredObservationResult = vendored_values.ObservationResult
    assert SourceObservation.__name__ == VendoredObservation.__name__ == "Observation"
    assert SourceObservationResult.__name__ == VendoredObservationResult.__name__ == "ObservationResult"


@pytest.mark.asyncio
async def test_logs_handle_reads_and_tails(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "agent.log").write_text("line1\nline2\nline3\n", encoding="utf-8")

    evidence = CandidateEvidence(descriptors={"logs": EvidenceDescriptor(kind="logs", format="dir", ref=str(log_dir))})
    handle = await evidence.logs("logs")
    assert handle is await evidence.logs("logs")  # cached
    assert await handle.list_files() == ["agent.log"]
    assert await handle.read_text("agent.log") == "line1\nline2\nline3\n"
    assert await handle.tail("agent.log", 2) == "line2\nline3"


def _both_format_evidence() -> CandidateEvidence:
    """A trial carrying both encodings of one trace, ATIF primary."""
    atif = EvidenceDescriptor(kind="trace", format=EVIDENCE_FORMAT_ATIF, data=_ATIF_TRAJECTORY)
    otlp = EvidenceDescriptor(kind="trace", format=EVIDENCE_FORMAT_OTLP, data={"resourceSpans": []})
    return CandidateEvidence(descriptors={"trace": atif, "trace:atif": atif, "trace:otlp": otlp})


@pytest.mark.asyncio
async def test_trace_format_selects_the_requested_encoding() -> None:
    evidence = _both_format_evidence()

    assert isinstance(await evidence.trace(), ATIFTraceHandle)
    assert isinstance(await evidence.trace(format=EVIDENCE_FORMAT_ATIF), ATIFTraceHandle)
    assert isinstance(await evidence.trace(format=EVIDENCE_FORMAT_OTLP), OTLPTraceHandle)


@pytest.mark.asyncio
async def test_trace_caches_each_format_separately() -> None:
    evidence = _both_format_evidence()

    atif = await evidence.trace(format=EVIDENCE_FORMAT_ATIF)
    otlp = await evidence.trace(format=EVIDENCE_FORMAT_OTLP)

    assert isinstance(atif, ATIFTraceHandle)
    assert isinstance(otlp, OTLPTraceHandle)
    assert atif is await evidence.trace(format=EVIDENCE_FORMAT_ATIF)
    assert otlp is await evidence.trace(format=EVIDENCE_FORMAT_OTLP)


@pytest.mark.asyncio
async def test_trace_shares_one_handle_when_the_primary_key_already_holds_the_format() -> None:
    evidence = CandidateEvidence(
        descriptors={"trace": EvidenceDescriptor(kind="trace", format=EVIDENCE_FORMAT_OTLP, data={"resourceSpans": []})}
    )

    handle = await evidence.trace(format=EVIDENCE_FORMAT_OTLP)
    assert handle is await evidence.trace()


@pytest.mark.asyncio
async def test_trace_treats_an_unset_format_as_atif_when_resolving() -> None:
    evidence = CandidateEvidence(descriptors={"trace": EvidenceDescriptor(kind="trace", data=_ATIF_TRAJECTORY)})

    assert isinstance(await evidence.trace(format=EVIDENCE_FORMAT_ATIF), ATIFTraceHandle)
    with pytest.raises(KeyError, match="in format 'otlp'"):
        await evidence.trace(format=EVIDENCE_FORMAT_OTLP)


@pytest.mark.asyncio
async def test_trace_rejects_a_format_no_descriptor_holds() -> None:
    evidence = CandidateEvidence(
        descriptors={"trace": EvidenceDescriptor(kind="trace", format=EVIDENCE_FORMAT_ATIF, data=_ATIF_TRAJECTORY)}
    )

    with pytest.raises(KeyError, match="missing evidence descriptor 'trace' in format 'otlp'"):
        await evidence.trace(format=EVIDENCE_FORMAT_OTLP)


@pytest.mark.asyncio
async def test_trace_ignores_a_format_qualified_key_whose_descriptor_disagrees() -> None:
    evidence = CandidateEvidence(
        descriptors={
            "trace": EvidenceDescriptor(kind="trace", format=EVIDENCE_FORMAT_ATIF, data=_ATIF_TRAJECTORY),
            "trace:otlp": EvidenceDescriptor(kind="trace", format=EVIDENCE_FORMAT_ATIF, data=_ATIF_TRAJECTORY),
        }
    )

    # Honouring the key suffix here would hand back an ATIFTraceHandle from a call
    # whose narrowed return type promises OTLPTraceHandle.
    with pytest.raises(KeyError, match="in format 'otlp'"):
        await evidence.trace(format=EVIDENCE_FORMAT_OTLP)
