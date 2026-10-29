# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo customization dataset prepare``, with the Files and Jobs services mocked."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import typer
from nhx.customization_common.cli.dataset_prepare import add_dataset_commands
from typer.testing import CliRunner, Result

_MODULE = "nhx.customization_common.cli.dataset_prepare"
_INSTRUCTION = [{"domain": "d", "instruction": f"q{i}", "response": f"a{i}"} for i in range(20)]
_PREFERENCE = [{"topic": "t", "instruction": f"q{i}", "chosen": "good", "rejected": "bad"} for i in range(20)]


@pytest.fixture(autouse=True)
def _platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NHX_BASE_URL", "http://nhx.test")


def _app() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def root() -> None: ...

    add_dataset_commands(app)
    return app


def _parquet(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _run(*args: str) -> tuple[Result, dict[str, dict[str, list[dict[str, Any]]]]]:
    """Run the command; return its result and, per fileset uploaded, each file's rows."""
    uploaded: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def create(files: object, *, name: str, source: Path, workspace: str, **_: object) -> str:
        uploaded[name] = {
            path.name: [json.loads(line) for line in path.read_text().splitlines()] for path in source.iterdir()
        }
        return f"{workspace}/{name}"

    with patch(f"{_MODULE}.create_dataset_fileset", side_effect=create):
        result = CliRunner().invoke(_app(), ["dataset", "prepare", *args])
    return result, uploaded


def test_local_parquet_instruction_rows_become_chat_splits(tmp_path: Path) -> None:
    source = _parquet(tmp_path / "data.parquet", _INSTRUCTION)

    result, uploaded = _run(str(source), "--name", "sft-data", "--format", "chat")

    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report == {
        "dataset": "default/sft-data",
        "test_dataset": "default/sft-data-test",
        "format": "chat",
        "trained_by": report["trained_by"],
        "columns": {"prompt": "instruction", "response": "response"},
        "rows": {"train": 16, "validation": 2, "test": 2},
    }
    assert report["trained_by"].startswith("automodel")
    assert sorted(uploaded["sft-data"]) == ["train.jsonl", "validation.jsonl"]
    assert sorted(uploaded["sft-data-test"]) == ["test.jsonl"]
    assert uploaded["sft-data"]["train.jsonl"][0]["messages"][1]["role"] == "assistant"


def test_preference_pairs_use_the_same_file_names_and_hold_out_no_test_by_default(tmp_path: Path) -> None:
    source = _parquet(tmp_path / "data.parquet", _PREFERENCE)

    result, uploaded = _run(str(source), "--name", "dpo-data", "--format", "preference")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["format"] == "preference"
    assert sorted(uploaded) == ["dpo-data"]
    assert sorted(uploaded["dpo-data"]) == ["train.jsonl", "validation.jsonl"]
    assert uploaded["dpo-data"]["train.jsonl"][0].keys() == {"prompt", "chosen", "rejected"}


def test_preference_pairs_need_a_validation_split(tmp_path: Path) -> None:
    source = _parquet(tmp_path / "data.parquet", _PREFERENCE)

    result, uploaded = _run(str(source), "--name", "dpo-data", "--format", "preference", "--val-fraction", "0")

    assert result.exit_code == 1
    assert "DPO needs a validation split" in result.output
    assert uploaded == {}


def test_the_same_seed_writes_the_same_split(tmp_path: Path) -> None:
    source = _parquet(tmp_path / "data.parquet", _INSTRUCTION)

    _, first = _run(str(source), "--name", "a", "--format", "chat", "--seed", "3")
    _, again = _run(str(source), "--name", "a", "--format", "chat", "--seed", "3")

    assert first == again


def test_a_fileset_is_downloaded_by_its_reference(tmp_path: Path) -> None:
    def download(files: object, *, remote_path: str, fileset: str, workspace: str, local_path: str) -> None:
        assert (workspace, fileset, remote_path) == ("ws", "dd-output", "dataset/")
        _parquet(Path(local_path) / "batch_0.parquet", _INSTRUCTION)

    with patch("filesets.transfer.download", side_effect=download):
        result, uploaded = _run("ws/dd-output#dataset/", "--name", "sft-data", "--format", "chat")

    assert result.exit_code == 0, result.output
    assert "train.jsonl" in uploaded["sft-data"]


def test_a_source_that_is_neither_local_nor_a_fileset_ref_fails() -> None:
    result, _ = _run("missing.parquet", "--name", "sft-data", "--format", "chat")

    assert result.exit_code == 1
    assert "is not a local path" in result.output


def test_dropped_rows_are_reported(tmp_path: Path) -> None:
    source = _parquet(tmp_path / "data.parquet", [*_INSTRUCTION, {"domain": "d", "instruction": "q", "response": None}])

    result, _ = _run(str(source), "--name", "sft-data", "--format", "chat")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["dropped"] == 1
    assert "Dropped 1 rows" in result.stderr


def test_the_format_is_required(tmp_path: Path) -> None:
    source = _parquet(tmp_path / "data.parquet", _INSTRUCTION)

    result, uploaded = _run(str(source), "--name", "sft-data")

    assert result.exit_code == 2
    assert "--format" in result.output
    assert uploaded == {}


def test_a_table_without_the_columns_of_the_format_fails_before_uploading(tmp_path: Path) -> None:
    source = _parquet(tmp_path / "data.parquet", _INSTRUCTION)

    result, uploaded = _run(str(source), "--name", "dpo-data", "--format", "preference")

    assert result.exit_code == 1
    assert "--format preference needs a chosen column" in result.output
    assert uploaded == {}


def test_named_columns_map_any_table(tmp_path: Path) -> None:
    rows = [{"q": f"q{i}", "pos": "p", "negs": ["n"]} for i in range(10)]
    source = _parquet(tmp_path / "data.parquet", rows)

    result, uploaded = _run(
        str(source),
        *("--name", "retrieval", "--format", "embedding", "--test-fraction", "0"),
        *("--query-column", "q", "--positive-column", "pos", "--negatives-column", "negs"),
    )

    assert result.exit_code == 0, result.output
    rows_written = uploaded["retrieval"]["train.jsonl"]
    assert len(rows_written) == 9
    assert all(row.keys() == {"query", "pos_doc", "neg_doc"} and row["neg_doc"] == ["n"] for row in rows_written)


def test_all_rows_empty_is_an_error_not_an_empty_dataset(tmp_path: Path) -> None:
    source = _parquet(tmp_path / "data.parquet", [{"instruction": "q", "response": None}] * 3)

    result, uploaded = _run(str(source), "--name", "sft-data", "--format", "chat")

    assert result.exit_code == 1
    assert "All 3 rows have an empty field" in result.output
    assert uploaded == {}
