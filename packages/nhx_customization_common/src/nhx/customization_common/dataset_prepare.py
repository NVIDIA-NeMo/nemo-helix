# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a table of rows, such as Data Designer output, into a Customizer dataset.

Reads Parquet, CSV or JSONL; maps the columns to the fields of the requested
:class:`DatasetFormat`; splits the rows with a seeded shuffle; and writes each split
as JSONL.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class InputFormat(StrEnum):
    PARQUET = "parquet"
    CSV = "csv"
    JSONL = "jsonl"


class DatasetFormat(StrEnum):
    """A row format Customizer trains on."""

    #: ``{"messages": [{"role", "content"}, ...]}``.
    CHAT = "chat"
    #: ``{"prompt", "chosen", "rejected"}``, for DPO.
    PREFERENCE = "preference"
    #: ``{"prompt", "completion"}``, trained as raw text with no chat template.
    PROMPT_COMPLETION = "prompt-completion"
    #: ``{"query", "pos_doc", "neg_doc": [...]}``, for retrieval.
    EMBEDDING = "embedding"
    #: ``{"text"}``, trained as raw text.
    TEXT = "text"


#: The backends that train on each format, and the job settings they need.
TRAINED_BY: dict[DatasetFormat, str] = {
    DatasetFormat.CHAT: "automodel (sft, distillation); unsloth with dataset.apply_chat_template: true",
    DatasetFormat.PREFERENCE: "rl with training.type: dpo",
    DatasetFormat.PROMPT_COMPLETION: "automodel; unsloth. Trains on raw text, with no chat template",
    DatasetFormat.EMBEDDING: "automodel with training.recipe: bi_encoder or cross_encoder",
    DatasetFormat.TEXT: "unsloth, with dataset.text_field: text",
}

#: The file each split is written to. Every backend reads these names.
SPLIT_FILES = {"train": "train.jsonl", "validation": "validation.jsonl", "test": "test.jsonl"}

_SUFFIXES = {InputFormat.PARQUET: ".parquet", InputFormat.CSV: ".csv", InputFormat.JSONL: ".jsonl"}

#: The flag that names the source column of each field.
FIELD_FLAGS = {
    "prompt": "--prompt-column",
    "response": "--response-column",
    "system": "--system-column",
    "messages": "--messages-column",
    "chosen": "--chosen-column",
    "rejected": "--rejected-column",
    "query": "--query-column",
    "positive": "--positive-column",
    "negatives": "--negatives-column",
    "text": "--text-column",
}

_PROMPT = ("instruction", "prompt")
_RESPONSE = ("response", "completion")

#: Per format, the fields every row needs, with the source columns tried when the caller names none.
_REQUIRED: dict[DatasetFormat, dict[str, tuple[str, ...]]] = {
    DatasetFormat.CHAT: {"prompt": _PROMPT, "response": _RESPONSE},
    DatasetFormat.PREFERENCE: {"prompt": _PROMPT, "chosen": ("chosen",), "rejected": ("rejected",)},
    DatasetFormat.PROMPT_COMPLETION: {"prompt": _PROMPT, "response": _RESPONSE},
    DatasetFormat.EMBEDDING: {"query": ("query",), "positive": ("pos_doc",), "negatives": ("neg_doc",)},
    DatasetFormat.TEXT: {"text": ("text",)},
}
#: Per format, the fields read only from a column the caller names.
_OPTIONAL: dict[DatasetFormat, tuple[str, ...]] = {
    DatasetFormat.CHAT: ("system",),
    DatasetFormat.PREFERENCE: ("system",),
}
#: Chat rows can instead come whole from a column of messages.
_MESSAGES = ("messages",)


class DatasetPrepareError(ValueError):
    """Raised when rows cannot be read, mapped or split."""


@dataclass(frozen=True)
class ColumnMapping:
    """The format to write, and the source column of each of its fields."""

    format: DatasetFormat
    columns: Mapping[str, str]


@dataclass(frozen=True)
class Splits:
    train: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    test: list[dict[str, Any]]


def read_rows(path: Path, input_format: InputFormat | None = None) -> list[dict[str, Any]]:
    """Read the rows of a Parquet, CSV or JSONL file, or of every such file in a directory.

    Without *input_format*, a file's extension gives its format, and a directory must hold
    files of one format. A directory, such as Data Designer's ``parquet-files``, is read file
    by file in name order, so the rows come out in the same order every time.
    """
    if path.is_dir():
        return _read_directory(path, input_format)
    return _read_file(path, input_format or _format_of(path))


def _read_directory(path: Path, input_format: InputFormat | None) -> list[dict[str, Any]]:
    suffixes = {suffix: fmt for fmt, suffix in _SUFFIXES.items()}
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in suffixes)
    if input_format is not None:
        files = [p for p in files if p.suffix == _SUFFIXES[input_format]]
    found = sorted({p.suffix for p in files})
    if not files:
        wanted = _SUFFIXES[input_format] if input_format else ", ".join(_SUFFIXES.values())
        raise DatasetPrepareError(f"{path} has no {wanted} file.")
    if len(found) > 1:
        raise DatasetPrepareError(f"{path} has {', '.join(found)} files. Pass --input-format to read one of them.")
    return [row for file in files for row in _read_file(file, suffixes[file.suffix])]


def _format_of(path: Path) -> InputFormat:
    for fmt, suffix in _SUFFIXES.items():
        if path.suffix == suffix:
            return fmt
    raise DatasetPrepareError(
        f"Cannot tell the format of {path} from its name. Pass --input-format ({', '.join(InputFormat)})."
    )


def _read_file(path: Path, input_format: InputFormat) -> list[dict[str, Any]]:
    if input_format is InputFormat.PARQUET:
        # Imported here: only Parquet needs it, and it is large.
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    if input_format is InputFormat.CSV:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    rows = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DatasetPrepareError(f"{path} line {number} is not JSON: {exc.msg}.") from exc
    return rows


def map_columns(
    dataset_format: DatasetFormat, columns: Iterable[str], named: Mapping[str, str | None] | None = None
) -> ColumnMapping:
    """Find the source column of each field *dataset_format* needs.

    A field takes the column the caller names in *named* (field to column), else the first
    of its default columns the table has, such as ``instruction`` or ``prompt`` for the
    prompt. A chat row comes either from a prompt and a response column, or whole from a
    ``messages`` column.
    """
    available = set(columns)
    named = {field: column for field, column in (named or {}).items() if column}
    allowed = {*_REQUIRED[dataset_format], *_OPTIONAL.get(dataset_format, ())}
    if dataset_format is DatasetFormat.CHAT:
        allowed.add("messages")
    if wrong := sorted(FIELD_FLAGS[field] for field in named if field not in allowed):
        raise DatasetPrepareError(f"{', '.join(wrong)} does not apply to --format {dataset_format}.")
    if missing := sorted(
        f"{column} ({FIELD_FLAGS[field]})" for field, column in named.items() if column not in available
    ):
        raise DatasetPrepareError(f"No column {', '.join(missing)}. {_columns(available)}")

    if dataset_format is DatasetFormat.CHAT and _uses_messages(named, available):
        return ColumnMapping(dataset_format, {"messages": named.get("messages") or _MESSAGES[0]})

    mapped = {field: column for field, column in named.items() if field in _OPTIONAL.get(dataset_format, ())}
    unmapped = []
    for field, defaults in _REQUIRED[dataset_format].items():
        column = named.get(field) or next((name for name in defaults if name in available), None)
        if column is None:
            unmapped.append(f"a {field} column ({' or '.join(defaults)}, or {FIELD_FLAGS[field]})")
        else:
            mapped[field] = column
    if unmapped:
        raise DatasetPrepareError(f"--format {dataset_format} needs {'; '.join(unmapped)}. {_columns(available)}")
    return ColumnMapping(dataset_format, mapped)


def _uses_messages(named: Mapping[str, str], available: set[str]) -> bool:
    """Whether chat rows come whole from a messages column rather than from prompt and response columns."""
    from_columns = {"prompt", "response", "system"} & named.keys()
    if "messages" in named:
        if from_columns:
            flags = ", ".join(FIELD_FLAGS[field] for field in sorted(from_columns))
            raise DatasetPrepareError(f"--messages-column cannot be combined with {flags}.")
        return True
    if from_columns or _MESSAGES[0] not in available:
        return False
    if all(any(name in available for name in defaults) for defaults in _REQUIRED[DatasetFormat.CHAT].values()):
        raise DatasetPrepareError(
            "The table has both a messages column and prompt and response columns. "
            "Pass --messages-column, or --prompt-column and --response-column."
        )
    return True


def _columns(available: set[str]) -> str:
    return f"Columns: {', '.join(sorted(available)) or 'none'}."


def to_training_rows(rows: Sequence[dict[str, Any]], mapping: ColumnMapping) -> tuple[list[dict[str, Any]], int]:
    """Convert *rows* to training rows; return them and how many were dropped for an empty field.

    A generated column can be empty where generation failed. Such a row cannot train,
    so it is dropped and counted rather than written. A cell of the wrong shape, such as
    messages that are not a list, is an error that names the row.
    """
    converted = []
    for number, row in enumerate(rows, start=1):
        training_row = _to_training_row(row, mapping, number)
        if training_row is not None:
            converted.append(training_row)
    return converted, len(rows) - len(converted)


def _to_training_row(row: dict[str, Any], mapping: ColumnMapping, number: int) -> dict[str, Any] | None:
    cells = {field: row.get(column) for field, column in mapping.columns.items()}
    if "messages" in cells:
        messages = _messages(cells["messages"], mapping.columns["messages"], number)
        return {"messages": messages} if messages else None
    if mapping.format is DatasetFormat.EMBEDDING:
        query, positive = _text(cells["query"]), _text(cells["positive"])
        negatives = _negatives(cells["negatives"], mapping.columns["negatives"], number)
        if not (_present(query, positive) and negatives):
            return None
        return {"query": query, "pos_doc": positive, "neg_doc": negatives}

    text = {field: _text(cell) for field, cell in cells.items()}
    if not _present(*(text[field] for field in _REQUIRED[mapping.format])):
        return None
    system = [{"role": "system", "content": text["system"]}] if text.get("system", "").strip() else []
    match mapping.format:
        case DatasetFormat.CHAT:
            user = {"role": "user", "content": text["prompt"]}
            return {"messages": [*system, user, {"role": "assistant", "content": text["response"]}]}
        case DatasetFormat.PREFERENCE:
            prompt = [*system, {"role": "user", "content": text["prompt"]}] if system else text["prompt"]
            return {"prompt": prompt, "chosen": text["chosen"], "rejected": text["rejected"]}
        case DatasetFormat.PROMPT_COMPLETION:
            return {"prompt": text["prompt"], "completion": text["response"]}
        case _:
            return {"text": text["text"]}


def _messages(cell: Any, column: str, number: int) -> list[dict[str, Any]] | None:
    """A cell of chat messages, which may be a JSON string; None when empty."""
    if isinstance(cell, str):
        if not cell.strip():
            return None
        try:
            cell = json.loads(cell)
        except json.JSONDecodeError as exc:
            raise DatasetPrepareError(f"Row {number}: {column} is not JSON: {exc.msg}.") from exc
    if not cell:
        return None
    if not isinstance(cell, list) or not all(
        isinstance(message, dict) and isinstance(message.get("role"), str) and isinstance(message.get("content"), str)
        for message in cell
    ):
        raise DatasetPrepareError(f"Row {number}: {column} is not a list of {{role, content}} messages.")
    return cell


def _negatives(cell: Any, column: str, number: int) -> list[str]:
    """A cell of negative documents, which may be a JSON string, without empty ones."""
    if isinstance(cell, str):
        try:
            cell = json.loads(cell) if cell.strip() else []
        except json.JSONDecodeError as exc:
            raise DatasetPrepareError(f"Row {number}: {column} is not a JSON list: {exc.msg}.") from exc
    if cell is None:
        return []
    if not isinstance(cell, list):
        raise DatasetPrepareError(f"Row {number}: {column} is not a list of documents.")
    return [text for text in (_text(document) for document in cell) if text.strip()]


def _text(value: Any) -> str:
    """A cell as text: structured columns, such as an LLM-structured one, as JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _present(*values: str) -> bool:
    return all(value.strip() for value in values)


def split_rows(rows: Sequence[dict[str, Any]], *, val_fraction: float, test_fraction: float, seed: int) -> Splits:
    """Shuffle *rows* with *seed* and cut validation and test off the front; the same seed gives the same split."""
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise DatasetPrepareError("The validation and test fractions must each be in [0, 1) and sum to less than 1.")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * test_fraction)
    n_val = round(len(shuffled) * val_fraction)
    test, validation, train = shuffled[:n_test], shuffled[n_test : n_test + n_val], shuffled[n_test + n_val :]
    if not train:
        raise DatasetPrepareError(f"{len(rows)} rows leave none to train on. Lower the validation and test fractions.")
    return Splits(train=train, validation=validation, test=test)


def write_split(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Write *rows* as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
