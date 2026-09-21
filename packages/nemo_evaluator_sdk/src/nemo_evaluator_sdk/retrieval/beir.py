# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load a local BEIR corpus without the heavyweight BEIR dependency."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "BeirCorpusDocument",
    "BeirDataset",
    "BeirDatasetError",
    "BeirQrels",
    "BeirQuery",
]

_CORPUS_FILE = Path("corpus.jsonl")
_QUERIES_FILE = Path("queries.jsonl")
_QRELS_FILE = Path("qrels/test.tsv")
_REQUIRED_FILES = (_CORPUS_FILE, _QUERIES_FILE, _QRELS_FILE)


class BeirDatasetError(ValueError):
    """Raised when a BEIR dataset has an invalid layout or record."""


@dataclass(frozen=True, slots=True)
class BeirCorpusDocument:
    """One document from ``corpus.jsonl``."""

    id: str
    text: str
    title: str = ""

    @property
    def content(self) -> str:
        """Return title and text in the form sent to the embedding model."""
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass(frozen=True, slots=True)
class BeirQuery:
    """One query from ``queries.jsonl``."""

    id: str
    text: str


@dataclass(frozen=True, slots=True)
class BeirQrels:
    """Unique ``(query-id, corpus-id)`` judgments from ``qrels/test.tsv``.

    Duplicate pairs keep the first score; ``dropped_rows`` is the number of
    later copies discarded.
    """

    judgments: dict[str, dict[str, int]]
    dropped_rows: int = 0


@dataclass(frozen=True, slots=True)
class BeirDataset:
    """An in-memory BEIR test split.

    ``from_path`` accepts either the BEIR directory itself or a fileset root
    containing the directory as ``eval_beir``.
    """

    root: Path
    corpus: dict[str, BeirCorpusDocument]
    queries: dict[str, BeirQuery]
    qrels: dict[str, dict[str, int]]
    dropped_qrel_rows: int = 0

    @classmethod
    def from_path(cls, path: str | Path) -> BeirDataset:
        """Load and validate a BEIR test split from ``path``."""
        root = _resolve_root(Path(path))
        corpus = _load_corpus(root / _CORPUS_FILE)
        queries = _load_queries(root / _QUERIES_FILE)
        qrels = _load_qrels(root / _QRELS_FILE)
        _validate_references(corpus, queries, qrels.judgments)
        return cls(
            root=root,
            corpus=corpus,
            queries=queries,
            qrels=qrels.judgments,
            dropped_qrel_rows=qrels.dropped_rows,
        )

    def query_rows(self) -> list[dict[str, Any]]:
        """Return one dataset row per judged query for metric scoring."""
        return [
            {
                "query_id": query_id,
                "query": self.queries[query_id].text,
                "qrels": self.qrels[query_id],
            }
            for query_id in self.qrels
        ]


def _resolve_root(path: Path) -> Path:
    candidates = (path, path / "eval_beir")
    for candidate in candidates:
        if all((candidate / relative_path).is_file() for relative_path in _REQUIRED_FILES):
            return candidate

    missing = [
        str(relative_path)
        for relative_path in _REQUIRED_FILES
        if not (path / relative_path).is_file() and not (path / "eval_beir" / relative_path).is_file()
    ]
    raise BeirDatasetError(f"{path} is not a BEIR test dataset; missing required files: {', '.join(missing)}")


def _load_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise BeirDatasetError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(value, dict):
                raise BeirDatasetError(f"{path}:{line_number}: expected a JSON object")
            records.append((line_number, value))
    if not records:
        raise BeirDatasetError(f"{path}: file contains no records")
    return records


def _required_string(record: dict[str, Any], field: str, path: Path, line_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise BeirDatasetError(f"{path}:{line_number}: field {field!r} must be a non-empty string")
    return value


def _load_corpus(path: Path) -> dict[str, BeirCorpusDocument]:
    corpus: dict[str, BeirCorpusDocument] = {}
    for line_number, record in _load_jsonl(path):
        document_id = _required_string(record, "_id", path, line_number)
        text = _required_string(record, "text", path, line_number)
        title = record.get("title", "")
        if not isinstance(title, str):
            raise BeirDatasetError(f"{path}:{line_number}: field 'title' must be a string")
        if document_id in corpus:
            raise BeirDatasetError(f"{path}:{line_number}: duplicate document id {document_id!r}")
        corpus[document_id] = BeirCorpusDocument(id=document_id, title=title, text=text)
    return corpus


def _load_queries(path: Path) -> dict[str, BeirQuery]:
    queries: dict[str, BeirQuery] = {}
    for line_number, record in _load_jsonl(path):
        query_id = _required_string(record, "_id", path, line_number)
        text = _required_string(record, "text", path, line_number)
        if query_id in queries:
            raise BeirDatasetError(f"{path}:{line_number}: duplicate query id {query_id!r}")
        queries[query_id] = BeirQuery(id=query_id, text=text)
    return queries


def _load_qrels(path: Path) -> BeirQrels:
    judgments: dict[str, dict[str, int]] = {}
    dropped_rows = 0
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        expected = ["query-id", "corpus-id", "score"]
        if reader.fieldnames != expected:
            raise BeirDatasetError(f"{path}: expected TSV header {expected}, got {reader.fieldnames}")
        for line_number, row in enumerate(reader, start=2):
            query_id = row["query-id"]
            document_id = row["corpus-id"]
            try:
                score = int(row["score"])
            except (TypeError, ValueError) as error:
                raise BeirDatasetError(f"{path}:{line_number}: score must be an integer") from error
            if not query_id or not document_id:
                raise BeirDatasetError(f"{path}:{line_number}: query-id and corpus-id must be non-empty")
            query_judgments = judgments.setdefault(query_id, {})
            if document_id in query_judgments:
                dropped_rows += 1
                continue
            query_judgments[document_id] = score
    if not judgments:
        raise BeirDatasetError(f"{path}: file contains no relevance judgments")
    if dropped_rows:
        logger.warning(
            "Dropped %s duplicate qrel row(s) from %s; kept the first score for each pair",
            dropped_rows,
            path,
        )
    return BeirQrels(judgments=judgments, dropped_rows=dropped_rows)


def _validate_references(
    corpus: dict[str, BeirCorpusDocument],
    queries: dict[str, BeirQuery],
    qrels: dict[str, dict[str, int]],
) -> None:
    unknown_queries = sorted(set(qrels) - set(queries))
    unknown_documents = sorted({document_id for judgments in qrels.values() for document_id in judgments} - set(corpus))
    if unknown_queries:
        raise BeirDatasetError(f"qrels reference unknown query ids: {', '.join(unknown_queries)}")
    if unknown_documents:
        raise BeirDatasetError(f"qrels reference unknown corpus ids: {', '.join(unknown_documents)}")
