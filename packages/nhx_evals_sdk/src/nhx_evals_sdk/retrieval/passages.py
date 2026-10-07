# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Title-aware passage truncation for embedding and ranking NIMs."""

from __future__ import annotations

from typing import Literal

from nhx_evals_sdk.retrieval.beir import BeirCorpusDocument

__all__ = ["DOCUMENT_CHARACTER_LIMIT", "Truncation", "passage_text"]

DOCUMENT_CHARACTER_LIMIT = 65535
Truncation = Literal["end", "start"]


def passage_text(
    document: BeirCorpusDocument,
    truncate_long_documents: Truncation | None = "end",
) -> str:
    """Return title plus text capped at the NIM 65535-character limit.

    Title is reserved first. Remaining budget is applied to ``text`` by keeping
    the start (``end`` truncation) or the tail (``start`` truncation). ``None``
    raises when the concatenated passage would exceed the limit.
    """
    title = document.title or None
    title_len = len(title) + 1 if title else 0
    text_limit = DOCUMENT_CHARACTER_LIMIT - title_len
    if text_limit <= 0:
        combined = title_len + len(document.text)
        if truncate_long_documents is None or title is None:
            raise ValueError(
                f"document {document.id!r} is too long ({combined} characters, the limit is "
                f"{DOCUMENT_CHARACTER_LIMIT}). Set truncate_long_documents to 'start' or 'end'."
            )
        return title[:DOCUMENT_CHARACTER_LIMIT]

    text = document.text
    if len(text) > text_limit:
        if truncate_long_documents == "end":
            text = text[:text_limit]
        elif truncate_long_documents == "start":
            text = text[-text_limit:]
        else:
            combined = len(title) + 1 + len(document.text) if title else len(document.text)
            raise ValueError(
                f"document {document.id!r} is too long ({combined} characters, the limit is "
                f"{DOCUMENT_CHARACTER_LIMIT}). Set truncate_long_documents to 'start' or 'end'."
            )

    return f"{title} {text}" if title else text
