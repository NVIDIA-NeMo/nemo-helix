# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from nhx.customization_common.dataset_prepare import (
    ColumnMapping,
    DatasetFormat,
    DatasetPrepareError,
    InputFormat,
    map_columns,
    read_rows,
    split_rows,
    to_training_rows,
)

# The columns of the Studio Data Designer templates.
_INSTRUCTION = {"domain": "math", "instruction": "What is 2+2?", "response": "4"}
_PREFERENCE = {"topic": "geo", "instruction": "Capital of France?", "chosen": "Paris.", "rejected": "Not sure."}
_MESSAGES = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]


def _parquet(path: Path, rows: list[dict]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _convert(dataset_format: DatasetFormat, rows: list[dict], **named: str) -> list[dict]:
    converted, _ = to_training_rows(rows, map_columns(dataset_format, rows[0].keys(), named))
    return converted


def test_reads_every_parquet_part_of_a_directory_in_name_order(tmp_path: Path) -> None:
    parts = tmp_path / "parquet-files"
    parts.mkdir()
    _parquet(parts / "batch_00001.parquet", [{"instruction": "b", "response": "2"}])
    _parquet(parts / "batch_00000.parquet", [{"instruction": "a", "response": "1"}])

    assert [row["instruction"] for row in read_rows(parts)] == ["a", "b"]


def test_reads_csv_and_jsonl(tmp_path: Path) -> None:
    (tmp_path / "rows.csv").write_text("instruction,response\nq,a\n")
    (tmp_path / "rows.jsonl").write_text(json.dumps(_INSTRUCTION) + "\n\n")

    assert read_rows(tmp_path / "rows.csv") == [{"instruction": "q", "response": "a"}]
    assert read_rows(tmp_path / "rows.jsonl") == [_INSTRUCTION]


def test_the_input_format_reads_a_file_whose_name_does_not_say(tmp_path: Path) -> None:
    (tmp_path / "rows").write_text(json.dumps(_INSTRUCTION) + "\n")

    with pytest.raises(DatasetPrepareError, match="Pass --input-format"):
        read_rows(tmp_path / "rows")
    assert read_rows(tmp_path / "rows", InputFormat.JSONL) == [_INSTRUCTION]


def test_a_directory_of_mixed_formats_needs_the_input_format(tmp_path: Path) -> None:
    _parquet(tmp_path / "a.parquet", [_INSTRUCTION])
    (tmp_path / "b.jsonl").write_text(json.dumps({**_INSTRUCTION, "domain": "jsonl"}) + "\n")

    with pytest.raises(DatasetPrepareError, match=r"has \.jsonl, \.parquet files\. Pass --input-format"):
        read_rows(tmp_path)
    assert read_rows(tmp_path, InputFormat.PARQUET) == [_INSTRUCTION]


def test_a_bad_jsonl_line_names_its_number(tmp_path: Path) -> None:
    (tmp_path / "rows.jsonl").write_text(json.dumps(_INSTRUCTION) + "\n{oops\n")

    with pytest.raises(DatasetPrepareError, match="line 2 is not JSON"):
        read_rows(tmp_path / "rows.jsonl")


def test_a_directory_without_rows_fails(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("x")

    with pytest.raises(DatasetPrepareError, match="has no"):
        read_rows(tmp_path)


@pytest.mark.parametrize(
    ("dataset_format", "columns", "expected"),
    [
        (DatasetFormat.CHAT, _INSTRUCTION, {"prompt": "instruction", "response": "response"}),
        (DatasetFormat.CHAT, ["prompt", "completion"], {"prompt": "prompt", "response": "completion"}),
        (DatasetFormat.CHAT, ["messages", "id"], {"messages": "messages"}),
        (
            DatasetFormat.PREFERENCE,
            _PREFERENCE,
            {"prompt": "instruction", "chosen": "chosen", "rejected": "rejected"},
        ),
        (DatasetFormat.PROMPT_COMPLETION, _INSTRUCTION, {"prompt": "instruction", "response": "response"}),
        (
            DatasetFormat.EMBEDDING,
            ["query", "pos_doc", "neg_doc"],
            {"query": "query", "positive": "pos_doc", "negatives": "neg_doc"},
        ),
        (DatasetFormat.TEXT, ["text"], {"text": "text"}),
    ],
)
def test_default_columns_map_without_flags(
    dataset_format: DatasetFormat, columns: Iterable[str], expected: dict[str, str]
) -> None:
    assert map_columns(dataset_format, columns) == ColumnMapping(dataset_format, expected)


def test_named_columns_override_the_defaults_field_by_field() -> None:
    mapping = map_columns(
        DatasetFormat.PREFERENCE,
        ["question", "good", "bad", "chosen", "rejected"],
        {"prompt": "question", "chosen": "good", "rejected": "bad"},
    )

    assert mapping.columns == {"prompt": "question", "chosen": "good", "rejected": "bad"}


def test_a_flag_for_another_format_is_refused() -> None:
    with pytest.raises(DatasetPrepareError, match="--chosen-column does not apply to --format chat"):
        map_columns(DatasetFormat.CHAT, _INSTRUCTION, {"chosen": "domain"})
    with pytest.raises(DatasetPrepareError, match="--system-column does not apply to --format prompt-completion"):
        map_columns(DatasetFormat.PROMPT_COMPLETION, _INSTRUCTION, {"system": "domain"})


def test_a_named_column_must_exist() -> None:
    with pytest.raises(DatasetPrepareError, match=r"No column answer \(--response-column\)\. Columns: "):
        map_columns(DatasetFormat.CHAT, _INSTRUCTION, {"response": "answer"})


def test_a_missing_field_names_its_defaults_and_flag() -> None:
    with pytest.raises(DatasetPrepareError) as error:
        map_columns(DatasetFormat.PREFERENCE, ["instruction", "chosen", "id"])

    assert str(error.value) == (
        "--format preference needs a rejected column (rejected, or --rejected-column). Columns: chosen, id, instruction."
    )


def test_messages_and_prompt_columns_together_are_ambiguous_until_named() -> None:
    columns = ["messages", "instruction", "response"]

    with pytest.raises(DatasetPrepareError, match="both a messages column and prompt and response columns"):
        map_columns(DatasetFormat.CHAT, columns)
    assert map_columns(DatasetFormat.CHAT, columns, {"messages": "messages"}).columns == {"messages": "messages"}
    assert map_columns(DatasetFormat.CHAT, columns, {"prompt": "instruction"}).columns == {
        "prompt": "instruction",
        "response": "response",
    }
    with pytest.raises(DatasetPrepareError, match="--messages-column cannot be combined with --prompt-column"):
        map_columns(DatasetFormat.CHAT, columns, {"messages": "messages", "prompt": "instruction"})


def test_chat_rows_with_a_system_column() -> None:
    rows = [{**_INSTRUCTION, "persona": "Be brief."}, {**_INSTRUCTION, "persona": ""}]

    converted = _convert(DatasetFormat.CHAT, rows, system="persona")

    assert converted == [
        {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        },
        {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]},
    ]


def test_messages_pass_through_and_may_be_a_json_string() -> None:
    rows = [{"conversation": _MESSAGES}, {"conversation": json.dumps(_MESSAGES)}, {"conversation": None}]

    converted, dropped = to_training_rows(
        rows, map_columns(DatasetFormat.CHAT, ["conversation"], {"messages": "conversation"})
    )

    assert converted == [{"messages": _MESSAGES}, {"messages": _MESSAGES}]
    assert dropped == 1


def test_messages_of_the_wrong_shape_name_the_row() -> None:
    mapping = map_columns(DatasetFormat.CHAT, ["messages"])

    with pytest.raises(DatasetPrepareError, match=r"Row 2: messages is not a list of \{role, content\} messages"):
        to_training_rows([{"messages": _MESSAGES}, {"messages": [{"text": "q"}]}], mapping)


def test_preference_rows_with_and_without_a_system_prompt() -> None:
    rows = [{**_PREFERENCE, "persona": "Be brief."}, {**_PREFERENCE, "persona": None}]

    converted = _convert(DatasetFormat.PREFERENCE, rows, system="persona")

    assert converted == [
        {
            "prompt": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Capital of France?"}],
            "chosen": "Paris.",
            "rejected": "Not sure.",
        },
        {"prompt": "Capital of France?", "chosen": "Paris.", "rejected": "Not sure."},
    ]


def test_prompt_completion_and_text_rows() -> None:
    assert _convert(DatasetFormat.PROMPT_COMPLETION, [_INSTRUCTION]) == [{"prompt": "What is 2+2?", "completion": "4"}]
    assert _convert(DatasetFormat.TEXT, [{"body": "Once upon a time."}], text="body") == [{"text": "Once upon a time."}]


def test_embedding_rows_take_negatives_as_a_list_or_a_json_list() -> None:
    rows = [
        {"query": "q", "pos_doc": "p", "neg_doc": ["n1", "", "n2"]},
        {"query": "q", "pos_doc": "p", "neg_doc": '["n3"]'},
        {"query": "q", "pos_doc": "p", "neg_doc": []},
    ]

    converted, dropped = to_training_rows(rows, map_columns(DatasetFormat.EMBEDDING, rows[0].keys()))

    assert converted == [
        {"query": "q", "pos_doc": "p", "neg_doc": ["n1", "n2"]},
        {"query": "q", "pos_doc": "p", "neg_doc": ["n3"]},
    ]
    assert dropped == 1


def test_embedding_negatives_that_are_not_a_list_name_the_row() -> None:
    rows = [{"query": "q", "pos_doc": "p", "neg_doc": "n1"}]

    with pytest.raises(DatasetPrepareError, match="Row 1: neg_doc is not a JSON list"):
        to_training_rows(rows, map_columns(DatasetFormat.EMBEDDING, rows[0].keys()))


def test_rows_with_an_empty_field_are_dropped_and_counted() -> None:
    rows = [_INSTRUCTION, {**_INSTRUCTION, "response": None}, {**_INSTRUCTION, "instruction": "  "}]

    converted, dropped = to_training_rows(rows, map_columns(DatasetFormat.CHAT, _INSTRUCTION))

    assert (len(converted), dropped) == (1, 2)


def test_a_structured_answer_is_written_as_json_and_text_is_kept_as_is() -> None:
    row = {"prompt": " q ", "completion": {"verdict": "phishing"}}

    [converted] = _convert(DatasetFormat.CHAT, [row])

    assert converted["messages"][0]["content"] == " q "
    assert converted["messages"][1]["content"] == '{"verdict": "phishing"}'


def test_the_split_is_deterministic_for_a_seed_and_covers_every_row() -> None:
    rows = [{"i": i} for i in range(100)]

    first = split_rows(rows, val_fraction=0.1, test_fraction=0.2, seed=7)
    again = split_rows(rows, val_fraction=0.1, test_fraction=0.2, seed=7)
    other = split_rows(rows, val_fraction=0.1, test_fraction=0.2, seed=8)

    assert first == again
    assert first != other
    assert (len(first.train), len(first.validation), len(first.test)) == (70, 10, 20)
    assert sorted(row["i"] for row in first.train + first.validation + first.test) == list(range(100))


@pytest.mark.parametrize(("val", "test"), [(-0.1, 0.0), (1.0, 0.0), (0.5, 0.5)])
def test_fractions_out_of_range_fail(val: float, test: float) -> None:
    with pytest.raises(DatasetPrepareError, match="fractions"):
        split_rows([{"i": 1}], val_fraction=val, test_fraction=test, seed=0)


def test_a_split_that_leaves_nothing_to_train_fails() -> None:
    with pytest.raises(DatasetPrepareError, match="none to train on"):
        split_rows([{"i": i} for i in range(3)], val_fraction=0.5, test_fraction=0.4, seed=0)


def test_a_split_asked_for_gets_a_row_even_from_a_small_table() -> None:
    """Rounding 5 rows x 0.1 would give 0; a split with a fraction above 0 still gets a row."""
    splits = split_rows([{"i": i} for i in range(5)], val_fraction=0.1, test_fraction=0.1, seed=0)
    assert (len(splits.train), len(splits.validation), len(splits.test)) == (3, 1, 1)

    no_test = split_rows([{"i": i} for i in range(5)], val_fraction=0.1, test_fraction=0.0, seed=0)
    assert (len(no_test.train), len(no_test.validation), len(no_test.test)) == (4, 1, 0)
