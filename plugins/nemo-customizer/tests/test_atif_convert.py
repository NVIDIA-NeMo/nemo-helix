# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from nemo_customizer.jobs.atif_convert import (
    AtifConvertJob,
    AtifConvertSpec,
    convert_filesets,
    fileset_prefix,
    stub_output,
    stub_row,
)
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.job_context import JobContext
from nhx.automodel.tasks.training.datasets.schemas import SFTPChatDatasetItemSchema

BUNDLE = "results/platform-job-attempt-A/agent-eval-results"
REF = "default/job-fileset-eval"
BASH = {
    "type": "function",
    "function": {"name": "bash", "description": "Run.", "parameters": {"type": "object", "properties": {}}},
}


def test_stub_rows_match_the_sft_chat_schema() -> None:
    first = stub_row()
    second = stub_row()
    second["messages"][1]["content"] = "What is 3+3?"
    SFTPChatDatasetItemSchema.model_validate(first)
    SFTPChatDatasetItemSchema.model_validate(second)
    output = stub_output(
        workspace="default",
        fileset="distill-data",
        node_key="prepare",
        trajectory_filesets=["default/job-fileset-eval"],
    )
    assert output["dataset"] == "default/distill-data#nodes/prepare/"
    assert output["trajectory_filesets"] == ["default/job-fileset-eval"]
    assert "tasks" not in output
    assert fileset_prefix("train[lora]") == "nodes/train_lora"


async def test_compile_requests_the_customizer_tasks_image(monkeypatch: pytest.MonkeyPatch) -> None:
    import nemo_customizer.jobs.atif_convert as module

    spec = AtifConvertSpec(workspace="default", fileset="distill-data", node_key="prepare")
    monkeypatch.setattr(
        module,
        "_tasks_image",
        lambda: ("example.invalid/nhx-customizer-tasks:test", ["/opt/venv/bin/python"]),
    )
    compiled = await AtifConvertJob.compile(
        workspace="default",
        spec=spec,
        entity_client=None,
        job_name="distill-t1",
        async_sdk=None,
    )
    step = compiled.steps[0]
    assert step.executor.provider == "cpu"
    assert step.executor.container.image == "example.invalid/nhx-customizer-tasks:test"
    assert step.executor.container.command == ["nemo_customizer.tasks.atif_convert"]
    assert step.config == {
        "workspace": "default",
        "fileset": "distill-data",
        "node_key": "prepare",
        "trajectory_filesets": [],
        "pass_filter": {},
        "validation_fraction": 0.1,
        "student_model": None,
        "max_tokens": None,
    }


def _atif(prompt: str, *replies: str, tools: bool = True) -> dict[str, Any]:
    """A trajectory: prompt, then one tool-calling agent step per reply, then a closing text reply."""
    steps: list[dict[str, Any]] = [{"source": "user", "message": prompt}]
    for number, reply in enumerate(replies):
        steps.append(
            {
                "source": "agent",
                "message": reply,
                "tool_calls": [
                    {"tool_call_id": f"c{number}", "function_name": "bash", "arguments": {"command": reply}}
                ],
                "observation": {"results": [{"source_call_id": f"c{number}", "content": f"out {number}"}]},
            }
        )
    steps.append({"source": "agent", "message": "done"})
    document: dict[str, Any] = {"schema_version": "ATIF-v1.7", "steps": steps}
    if tools:
        document["agent"] = {"tool_definitions": [BASH]}
    return document


def _trial(trial_id: str, task_id: str, *, ref: str | None = "", status: str = "completed") -> dict[str, Any]:
    ref = f"evidence/{trial_id}/traces/{trial_id}.atif.json" if ref == "" else ref
    descriptors = {"trace:atif": {"kind": "trace", "format": "atif", "ref": ref}} if ref else {}
    return {"id": trial_id, "task_id": task_id, "status": status, "evidence": {"descriptors": descriptors}}


def _score(trial_id: str, reward: float) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "status": "completed",
        "metric_type": "harbor_reward",
        "outputs": [{"name": "reward", "value": reward}],
    }


class MemorySource:
    """A fileset as `{path: bytes}`, shaped like the Files API."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def list_paths(self, ref: str) -> list[str]:
        assert ref == REF
        return sorted(self.files)

    def read(self, ref: str, path: str) -> bytes:
        assert ref == REF
        try:
            return self.files[path]
        except KeyError:
            response = httpx.Response(404, request=httpx.Request("GET", f"http://platform/{path}"))
            raise NotFoundError(response) from None


def _bundle(
    trials: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    traces: Mapping[str, dict[str, Any] | bytes],
    *,
    bundle: str = BUNDLE,
) -> dict[str, bytes]:
    files = {
        f"{bundle}/trials.jsonl": "".join(json.dumps(t) + "\n" for t in trials).encode(),
        f"{bundle}/scores.jsonl": "".join(json.dumps(s) + "\n" for s in scores).encode(),
    }
    for name, content in traces.items():
        files[f"{bundle}/evidence/{name}/traces/{name}.atif.json"] = (
            content if isinstance(content, bytes) else json.dumps(content).encode()
        )
    return files


def _spec(**overrides: Any) -> AtifConvertSpec:
    values: dict[str, Any] = {
        "workspace": "default",
        "fileset": "distill-data",
        "node_key": "prepare",
        "trajectory_filesets": [REF],
    }
    return AtifConvertSpec(**{**values, **overrides})


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_convert_filesets_writes_a_row_per_assistant_turn_split_by_task(tmp_path: Path) -> None:
    tasks = [f"task-{i}" for i in range(10)]
    trials = [_trial(f"t{i}", task) for i, task in enumerate(tasks)]
    traces = {f"t{i}": _atif(f"Fix task {i}", "ls", "cat") for i in range(10)}
    source = MemorySource(_bundle(trials, [_score(t["id"], 1.0) for t in trials], traces))

    result = convert_filesets(source, _spec(), tmp_path)

    train, validation = _read(result.train_path), _read(result.validation_path)
    # Three agent turns per trajectory (two tool calls and a closing reply).
    assert len(train) + len(validation) == 30
    assert result.output["examples"] == len(train) == 27
    assert result.output["validation_examples"] == len(validation) == 3
    assert result.output["runs"] == 10
    assert result.output["trials_seen"] == 10
    assert result.output["skipped"] == {}
    assert result.output["validation_from_train"] is False
    assert result.output["dataset"] == "default/distill-data#nodes/prepare/"
    for row in [*train, *validation]:
        SFTPChatDatasetItemSchema.model_validate(row)
        assert row["messages"][-1]["role"] == "assistant"
    # No task straddles the split: that would leak held-out prompts into training.
    held_out = set(result.output["held_out"])
    assert len(held_out) == 1
    prompts = {row["messages"][0]["content"] for row in validation}
    assert prompts == {f"Fix task {tasks.index(next(iter(held_out)))}"}
    assert prompts.isdisjoint({row["messages"][0]["content"] for row in train})


def test_pass_filter_keeps_only_passing_trials(tmp_path: Path) -> None:
    trials = [_trial("good", "task-a"), _trial("bad", "task-b"), _trial("unscored", "task-c")]
    scores = [_score("good", 1.0), _score("bad", 0.0)]
    traces = {name: _atif(f"prompt {name}", "ls") for name in ("good", "bad", "unscored")}
    source = MemorySource(_bundle(trials, scores, traces))

    result = convert_filesets(
        source,
        _spec(
            pass_filter={"metric_type": "harbor_reward", "output": "reward", "min_value": 1.0}, validation_fraction=0
        ),
        tmp_path,
    )

    assert result.output["runs"] == 1
    assert result.output["skipped"] == {"filtered_by_score": 2}
    assert {row["messages"][0]["content"] for row in _read(result.train_path)} == {"prompt good"}


def test_every_unusable_trial_is_counted_not_fatal(tmp_path: Path) -> None:
    trials = [
        _trial("ok", "t1"),
        _trial("failed", "t2", status="failed"),
        _trial("no-trace", "t3", ref=None),
        _trial("missing", "t4", ref="evidence/missing/traces/missing.atif.json"),
        _trial("corrupt", "t5"),
        _trial("not-an-object", "t6"),
        _trial("no-tools", "t7"),
    ]
    traces: dict[str, dict[str, Any] | bytes] = {
        "ok": _atif("p", "ls"),
        "failed": _atif("p", "ls"),
        "corrupt": b"{not json",
        "not-an-object": b"[1]",
        "no-tools": _atif("p", "ls", tools=False),
    }
    source = MemorySource(_bundle(trials, [], traces))

    result = convert_filesets(source, _spec(validation_fraction=0), tmp_path)

    assert result.output["runs"] == 1
    assert result.output["trials_seen"] == 7
    assert result.output["skipped"] == {
        "no_atif_trace": 1,
        "no_tool_definitions": 1,
        "trace_invalid": 2,
        "trace_missing": 1,
        "trial_not_completed": 1,
    }


def test_a_retried_attempt_does_not_duplicate_trials(tmp_path: Path) -> None:
    trial = _trial("t1", "task-a")
    files = _bundle([trial], [], {"t1": _atif("p", "ls")})
    files.update(
        _bundle([trial], [], {"t1": _atif("p", "ls")}, bundle="results/platform-job-attempt-B/agent-eval-results")
    )

    result = convert_filesets(MemorySource(files), _spec(validation_fraction=0), tmp_path)

    assert result.output["trials_seen"] == 1
    assert result.output["examples"] == 2


def test_a_single_task_validates_on_the_training_rows_and_says_so(tmp_path: Path) -> None:
    source = MemorySource(_bundle([_trial("t1", "only")], [], {"t1": _atif("p", "ls")}))

    result = convert_filesets(source, _spec(), tmp_path)

    assert result.output["validation_from_train"] is True
    assert result.output["held_out"] == []
    assert result.output["validation_examples"] == result.output["examples"] == 2
    assert _read(result.validation_path) == _read(result.train_path)


def test_no_results_bundle_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no trials.jsonl"):
        convert_filesets(MemorySource({"logs/x.parquet": b""}), _spec(), tmp_path)


def test_no_training_rows_is_an_error_that_explains_why(tmp_path: Path) -> None:
    source = MemorySource(_bundle([_trial("t1", "a", status="failed")], [], {}))
    with pytest.raises(RuntimeError, match="trial_not_completed"):
        convert_filesets(source, _spec(), tmp_path)


def test_unknown_pass_filter_keys_fail_before_any_work(tmp_path: Path) -> None:
    source = MemorySource(_bundle([_trial("t1", "a")], [], {"t1": _atif("p")}))
    with pytest.raises(ValueError):
        convert_filesets(source, _spec(pass_filter={"metrik": "x"}), tmp_path)


def test_run_streams_the_dataset_and_writes_output_last(monkeypatch: pytest.MonkeyPatch) -> None:
    import nemo_customizer.jobs.atif_convert as module

    trials = [_trial(f"t{i}", f"task-{i}") for i in range(4)]
    traces = {f"t{i}": _atif(f"prompt {i}", "ls") for i in range(4)}
    served = MemorySource(_bundle(trials, [], traces))
    uploaded: dict[str, bytes] = {}
    order: list[str] = []

    class Files:
        def upload_file(self, *, workspace: str, name: str, path: str, content: Any) -> None:
            assert (workspace, name) == ("default", "distill-data")
            uploaded[path] = content if isinstance(content, bytes) else b"".join(content)
            order.append(path)

    class SDK:
        pass

    monkeypatch.setattr(module.FilesClient, "from_client", lambda client: Files())
    monkeypatch.setattr(module, "FilesClientSource", lambda files, workspace: served)
    # `MemorySource` asserts the ref, so the job must pass the spec's ref through unchanged.
    result = AtifConvertJob().run(
        {
            "workspace": "default",
            "fileset": "distill-data",
            "node_key": "prepare",
            "trajectory_filesets": [REF],
            "validation_fraction": 0.25,
        },
        ctx=cast(JobContext, None),
        sdk=cast(NemoClient, SDK()),
    )

    assert result == {"status": "completed", "dataset": "default/distill-data#nodes/prepare/"}
    assert order[-1] == "nodes/prepare/output.json"  # the controller treats output.json as "dataset complete"
    assert set(order) == {"nodes/prepare/train.jsonl", "nodes/prepare/validation.jsonl", "nodes/prepare/output.json"}
    output = json.loads(uploaded["nodes/prepare/output.json"])
    train = [json.loads(line) for line in uploaded["nodes/prepare/train.jsonl"].decode().splitlines()]
    validation = [json.loads(line) for line in uploaded["nodes/prepare/validation.jsonl"].decode().splitlines()]
    assert output["examples"] == len(train) == 6
    assert output["validation_examples"] == len(validation) == 2
    assert output["trajectory_filesets"] == [REF]
    assert len(output["held_out"]) == 1
    for row in [*train, *validation]:
        SFTPChatDatasetItemSchema.model_validate(row)


def test_run_requires_filesets_and_a_client() -> None:
    config = {"workspace": "default", "fileset": "distill-data", "node_key": "prepare"}
    with pytest.raises(RuntimeError, match="evaluation output filesets"):
        AtifConvertJob().run(config, ctx=cast(JobContext, None), sdk=cast(NemoClient, object()))
    with pytest.raises(RuntimeError, match="platform client"):
        AtifConvertJob().run({**config, "trajectory_filesets": [REF]}, ctx=cast(JobContext, None), sdk=None)


class CharCounter:
    """One token per character of the serialized row."""

    def count(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> int:
        return len(json.dumps(messages)) + len(json.dumps(tools or []))


def _long_and_short_bundle() -> MemorySource:
    """Two tasks: a long trajectory (6 tool turns, big outputs) and a short one."""
    long_steps = _atif("long task", *["x" * 400] * 6)
    short_steps = _atif("short task", "ls")
    trials = [_trial("long", "task-long"), _trial("short", "task-short")]
    return MemorySource(_bundle(trials, [], {"long": long_steps, "short": short_steps}))


def test_a_length_limit_needs_a_student_model_but_a_disabled_one_does_not() -> None:
    with pytest.raises(ValueError, match="student_model"):
        _spec(max_tokens={"tokens": "32k"})
    disabled = _spec(max_tokens={"enabled": False, "tokens": "32k"})
    assert disabled.budget is None
    enabled = _spec(student_model="default/qwen3-5-4b", max_tokens={"behavior": "drop-longest", "tokens": "32k"})
    assert enabled.budget is not None and enabled.budget.tokens == 32768


def test_the_limit_needs_a_counter(tmp_path: Path) -> None:
    spec = _spec(student_model="default/s", max_tokens={"tokens": 100})
    with pytest.raises(ValueError, match="token counter"):
        convert_filesets(_long_and_short_bundle(), spec, tmp_path)


@pytest.mark.parametrize(
    ("behavior", "expect"),
    [
        # long chain (6 turns + final reply = 7 rows): rows past the limit are handled per behavior
        ("drop-chain", {"chains": 1, "train_trials": {"short"}, "truncated": 0}),
        ("drop-longest", {"chains": 0, "train_trials": {"long", "short"}, "truncated": 0}),
        ("truncate-earlier-turns", {"chains": 0, "train_trials": {"long", "short"}, "truncated": None}),
    ],
)
def test_each_length_behavior_end_to_end(tmp_path: Path, behavior: str, expect: dict[str, Any]) -> None:
    source = _long_and_short_bundle()
    limit = 2500
    spec = _spec(
        validation_fraction=0,
        student_model="default/s",
        max_tokens={"behavior": behavior, "tokens": limit},
    )

    result = convert_filesets(source, spec, tmp_path, CharCounter())

    rows = _read(result.train_path)
    prompts = {row["messages"][0]["content"] for row in rows}
    assert prompts == {f"{name} task" for name in expect["train_trials"]}
    assert all(
        CharCounter().count(r["messages"], r.get("tools")) <= limit for r in rows
    )  # nothing over the limit ships
    output = result.output
    assert output["max_tokens"] == {"behavior": behavior, "tokens": limit}
    assert output["skipped"].get("chain_over_max_tokens", 0) == expect["chains"]
    if behavior == "drop-chain":
        assert output["dropped_trials"] == ["long"]
        assert output["skipped"]["row_over_max_tokens"] == 7  # the whole chain went
        assert output["truncated_rows"] == 0
    elif behavior == "drop-longest":
        assert output["skipped"]["row_over_max_tokens"] >= 1
        assert output["truncated_rows"] == 0
        assert output["dropped_trials"] == []  # the chain survives in part
    else:
        assert output["truncated_rows"] >= 1
        assert output["skipped"].get("row_over_max_tokens", 0) == 0
        # truncation keeps every turn as a row: 7 for the long trajectory, 2 for the short one
        assert output["examples"] == 9


def test_without_a_limit_the_output_reports_none_and_zero(tmp_path: Path) -> None:
    result = convert_filesets(_long_and_short_bundle(), _spec(validation_fraction=0), tmp_path)
    assert result.output["max_tokens"] is None
    assert result.output["truncated_rows"] == 0
    assert result.output["dropped_trials"] == []
    assert "row_over_max_tokens" not in result.output["skipped"]


def test_a_limit_that_removes_everything_fails_and_says_why(tmp_path: Path) -> None:
    spec = _spec(student_model="default/s", max_tokens={"tokens": 10})
    with pytest.raises(RuntimeError, match="chain_over_max_tokens"):
        convert_filesets(_long_and_short_bundle(), spec, tmp_path, CharCounter())


def test_run_loads_the_student_tokenizer_before_reading_any_trajectory(monkeypatch: pytest.MonkeyPatch) -> None:
    import nemo_customizer.jobs.atif_convert as module

    touched: list[str] = []

    class Files:
        def upload_file(self, **kwargs: Any) -> None:
            touched.append("upload")

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("no tokenizer_config.json")

    monkeypatch.setattr(module.FilesClient, "from_client", lambda client: Files())
    monkeypatch.setattr(module, "load_student_counter", fail)
    monkeypatch.setattr(module, "FilesClientSource", lambda *a: touched.append("source") or _long_and_short_bundle())

    with pytest.raises(RuntimeError, match="tokenizer_config"):
        AtifConvertJob().run(
            {
                "workspace": "default",
                "fileset": "distill-data",
                "node_key": "prepare",
                "trajectory_filesets": [REF],
                "student_model": "default/qwen3-5-4b",
                "max_tokens": {"tokens": "32k"},
            },
            ctx=cast(JobContext, None),
            sdk=cast(NemoClient, object()),
        )
    assert touched == []


def _listing(paths: list[str]) -> Any:
    """What `FilesClient.list_files` returns: a response whose `.data()` is a listing with a `.data` list."""
    listing = type("Listing", (), {"data": [type("Item", (), {"path": path})() for path in paths]})()
    return type("Response", (), {"data": lambda self: listing})()


def _student_clients(
    monkeypatch: pytest.MonkeyPatch, *, fileset: str | None, files: dict[str, bytes], trust_remote_code: bool = False
) -> list[str]:
    import nemo_customizer.jobs.atif_convert as module

    downloaded: list[str] = []

    class Models:
        def get_model(self, *, name: str, workspace: str) -> Any:
            assert (workspace, name) == ("default", "qwen3-5-4b")
            model = type("Model", (), {"fileset": fileset, "trust_remote_code": trust_remote_code})()
            return type("Response", (), {"data": lambda self: model})()

    class Files:
        def list_files(self, *, workspace: str, name: str) -> Any:
            assert (workspace, name) == ("default", "qwen3-5-4b-weights")
            return _listing(list(files))

        def download_file(self, *, workspace: str, name: str, path: str) -> Any:
            downloaded.append(path)
            return type("Body", (), {"read": lambda self: files[path]})()

    monkeypatch.setattr(module.ModelsClient, "from_client", lambda client: Models())
    monkeypatch.setattr(module.FilesClient, "from_client", lambda client: Files())
    return downloaded


def test_the_student_counter_reads_only_tokenizer_files_from_the_models_fileset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import nemo_customizer.jobs.atif_convert as module

    weights = {
        "model-00001-of-00002.safetensors": b"w" * 10,
        "model.safetensors.index.json": b"{}",
        "tokenizer_config.json": b'{"a": 1}',
        "tokenizer.json": b"{}",
        "chat_template.jinja": b"{{ x }}",
        "config.json": b"{}",
        "modeling_custom.py": b"print('remote code')",
        "nested/tokenizer.json": b"{}",
    }
    downloaded = _student_clients(monkeypatch, fileset="default/qwen3-5-4b-weights", files=weights)
    loaded: list[Any] = []

    def load_tokenizer(directory: Path, *, trust_remote_code: bool) -> str:
        loaded.append((directory, trust_remote_code))
        return "tokenizer"

    monkeypatch.setattr(module, "_load_tokenizer", load_tokenizer)

    counter = module.load_student_counter(cast(NemoClient, object()), "default/qwen3-5-4b", "default", tmp_path)

    assert sorted(downloaded) == ["chat_template.jinja", "config.json", "tokenizer.json", "tokenizer_config.json"]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(downloaded)  # weights, remote code, subdirs untouched
    assert loaded == [(tmp_path, False)]
    assert isinstance(counter, module.ChatTemplateCounter)


def test_remote_code_files_are_fetched_only_when_the_model_trusts_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import nemo_customizer.jobs.atif_convert as module

    files = {"tokenizer_config.json": b"{}", "tokenization_custom.py": b"x = 1"}
    downloaded = _student_clients(
        monkeypatch, fileset="default/qwen3-5-4b-weights", files=files, trust_remote_code=True
    )
    trust: list[bool] = []
    monkeypatch.setattr(
        module, "_load_tokenizer", lambda directory, *, trust_remote_code: trust.append(trust_remote_code)
    )

    module.load_student_counter(cast(NemoClient, object()), "default/qwen3-5-4b", "default", tmp_path)

    assert sorted(downloaded) == ["tokenization_custom.py", "tokenizer_config.json"]
    assert trust == [True]


@pytest.mark.parametrize(
    ("fileset", "files", "message"),
    [
        (None, {}, "no weights fileset"),
        ("default/qwen3-5-4b-weights", {"model.safetensors": b""}, "no tokenizer_config.json"),
        ("https://example.invalid/a/b/c", {}, "unsupported fileset"),
    ],
)
def test_the_student_counter_fails_clearly_when_it_cannot_find_a_tokenizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fileset: str | None, files: dict[str, bytes], message: str
) -> None:
    import nemo_customizer.jobs.atif_convert as module

    _student_clients(monkeypatch, fileset=fileset, files=files)
    with pytest.raises(RuntimeError, match=message):
        module.load_student_counter(cast(NemoClient, object()), "default/qwen3-5-4b", "default", tmp_path)


def test_files_client_source_lists_and_reads_through_the_files_api_shapes() -> None:
    from nemo_customizer.jobs.atif_convert import FilesClientSource

    calls: list[tuple[str, str, str | None]] = []

    class Files:
        def list_files(self, *, workspace: str, name: str) -> Any:
            calls.append((workspace, name, None))
            return _listing(["results/a/agent-eval-results/trials.jsonl", "logs/x.parquet"])

        def download_file(self, *, workspace: str, name: str, path: str) -> Any:
            calls.append((workspace, name, path))
            return type("Body", (), {"read": lambda self: b"payload"})()

    source = FilesClientSource(cast(Any, Files()), "wf-workspace")

    assert source.list_paths("other/job-fileset-eval") == [
        "results/a/agent-eval-results/trials.jsonl",
        "logs/x.parquet",
    ]
    assert source.read("job-fileset-eval", "a/b.json") == b"payload"
    # A ref carries its own workspace; a bare name falls back to the workflow workspace.
    assert calls == [("other", "job-fileset-eval", None), ("wf-workspace", "job-fileset-eval", "a/b.json")]
