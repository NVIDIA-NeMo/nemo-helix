# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert wrapped Automodel retrieval JSON into inline JSONL for Customizer."""

import json
import logging
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ADDITIONAL_DIRNAME = "additional"
_ROOT_NAMES = frozenset({"training.jsonl", "eval_beir", ADDITIONAL_DIRNAME})


@dataclass
class _CorpusResolver:
    """Resolve document entries to text, counting ids the corpus cannot resolve."""

    corpus_by_id: dict[str, str]
    unresolved: int = 0

    def resolve_all(self, docs: Iterable[Any]) -> list[str]:
        resolved: list[str] = []
        for doc in docs:
            text = self._resolve(doc)
            if text is None:
                self.unresolved += 1
            else:
                resolved.append(text)
        return resolved

    def _resolve(self, doc: Any) -> str | None:
        if isinstance(doc, str):
            return _usable_text(doc)
        if not isinstance(doc, dict):
            return None
        for key in ("text", "contents"):
            text = _usable_text(doc.get(key)) if key in doc else None
            if text is not None:
                return text
        doc_id = doc.get("id")
        if doc_id is None:
            return None
        return self.corpus_by_id.get(str(doc_id))


def wrapped_to_inline_jsonl(train_json: Path, output_jsonl: Path, corpus_parquet: Path | None = None) -> Path:
    """Emit ``{query, pos_doc, neg_doc}`` JSONL from wrapped Automodel JSON."""
    payload = json.loads(train_json.read_text(encoding="utf-8"))
    resolver = _CorpusResolver(_load_corpus(train_json, corpus_parquet))
    dropped_records = 0

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in payload.get("data", []):
            rows = _inline_rows(record, resolver)
            if not rows:
                dropped_records += 1
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if resolver.unresolved or dropped_records:
        logger.warning(
            "Skipped %s unresolved corpus document(s); dropped %s unusable record(s)",
            resolver.unresolved,
            dropped_records,
        )
    return output_jsonl


def move_aux_files_to_additional(output_dir: Path) -> None:
    """Move everything except ``training.jsonl`` and ``eval_beir/`` under ``additional/``.

    Mining and conversion write wrapped JSON, corpus parquet, and embedding caches
    beside the files Automodel and retrieve-eval read. Those extras stay in the
    artifacts result, nested so a fileset whose root is this directory is
    immediately usable as ``dataset.training``.
    """
    extra = output_dir / ADDITIONAL_DIRNAME
    extra.mkdir(parents=True, exist_ok=True)
    for child in list(output_dir.iterdir()):
        if child.name in _ROOT_NAMES:
            continue
        dest = extra / child.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(child), str(dest))


def _inline_rows(record: dict[str, Any], resolver: _CorpusResolver) -> list[dict[str, Any]]:
    """Inline rows for one wrapped record, empty when unresolved ids made it unusable."""
    requested_negatives = record.get("neg_doc") or []
    positives = resolver.resolve_all(record.get("pos_doc") or [])
    negatives = resolver.resolve_all(requested_negatives)

    if not positives:
        return []
    # Keeping a mined record whose negatives all failed to resolve would train it as
    # positives-only, which silently changes the loss the recipe expects.
    if requested_negatives and not negatives:
        return []

    query = record.get("question", "")
    return [{"query": query, "pos_doc": positive, "neg_doc": negatives} for positive in positives]


def _load_corpus(train_json: Path, corpus_parquet: Path | None) -> dict[str, str]:
    parquet = corpus_parquet or train_json.parent / "corpus" / "train.parquet"
    if parquet.suffix != ".parquet" or not parquet.exists():
        return {}
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Reading a retrieval corpus parquet requires pandas. "
            "cpu-tasks and nmp-automodel-training already include it."
        ) from exc
    frame = pd.read_parquet(parquet)
    id_column = "id" if "id" in frame.columns else frame.columns[0]
    text_column = next((column for column in ("text", "contents") if column in frame.columns), None)
    if text_column is None:
        return {}
    corpus_by_id: dict[str, str] = {}
    for doc_id, text in zip(frame[id_column], frame[text_column], strict=True):
        usable = _usable_text(text)
        if usable is None:
            continue
        corpus_by_id[str(doc_id)] = usable
    return corpus_by_id


def _usable_text(value: Any) -> str | None:
    """Return document text only when it is a non-empty string."""
    if isinstance(value, str) and value.strip():
        return value
    return None
