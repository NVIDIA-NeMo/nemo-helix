# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx_evals_sdk.retrieval.beir import BeirCorpusDocument
from nhx_evals_sdk.retrieval.passages import DOCUMENT_CHARACTER_LIMIT, passage_text


def test_title_aware_end_truncation_caps_concatenated_passage() -> None:
    title = "Title"
    head = "HEAD"
    tail = "TAIL"
    text = head + ("x" * DOCUMENT_CHARACTER_LIMIT) + tail
    passage = passage_text(BeirCorpusDocument(id="d1", title=title, text=text), "end")

    assert len(passage) == DOCUMENT_CHARACTER_LIMIT
    assert passage.startswith(f"{title} {head}")
    assert tail not in passage


def test_start_truncation_keeps_the_tail() -> None:
    title = "T"
    text = "head" + "z" * DOCUMENT_CHARACTER_LIMIT
    passage = passage_text(BeirCorpusDocument(id="d1", title=title, text=text), "start")

    assert len(passage) == DOCUMENT_CHARACTER_LIMIT
    assert passage.startswith("T ")
    assert passage.endswith("z")
    assert "head" not in passage


def test_unset_truncation_raises_for_oversize_title_plus_text() -> None:
    document = BeirCorpusDocument(id="d1", title="Title", text="x" * DOCUMENT_CHARACTER_LIMIT)
    with pytest.raises(ValueError, match="too long"):
        passage_text(document, None)


def test_unset_truncation_raises_when_title_alone_exceeds_the_limit() -> None:
    document = BeirCorpusDocument(id="d1", title="T" * (DOCUMENT_CHARACTER_LIMIT + 1), text="body")
    with pytest.raises(ValueError, match="too long"):
        passage_text(document, None)
