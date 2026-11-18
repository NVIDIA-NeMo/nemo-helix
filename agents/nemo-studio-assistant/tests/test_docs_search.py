# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from nemo_studio_assistant import docs_search

pytestmark = pytest.mark.unit


@pytest.fixture
def docs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    root = tmp_path / "docs-snapshot"
    (root / "guardrails").mkdir(parents=True)
    (root / "guardrails" / "index.mdx").write_text(
        "---\n"
        'title: "About Guardrails"\n'
        "---\n"
        "import { Callout } from 'components'\n"
        "Guardrails check messages against a policy.\n\n"
        "## Attach to a VirtualModel\n\n"
        "Add the nemo-guardrails middleware to a VirtualModel so every request is checked.\n",
        encoding="utf-8",
    )
    (root / "inference.md").write_text(
        "# Inference providers\n\nConnect build.nvidia.com as an inference provider.\n",
        encoding="utf-8",
    )
    (root / "notes.txt").write_text("guardrails virtualmodel", encoding="utf-8")
    monkeypatch.setenv(docs_search.DOCS_DIR_ENV, str(root))
    docs_search._load_index.cache_clear()
    yield root
    docs_search._load_index.cache_clear()


def test_search_ranks_the_matching_section_first(docs_root: Path) -> None:
    results = docs_search.search("attach guardrail virtualmodel")

    assert results[0]["path"] == "guardrails/index.mdx"
    assert results[0]["title"] == "About Guardrails"
    assert results[0]["section"] == "Attach to a VirtualModel"
    assert "nemo-guardrails middleware" in str(results[0]["excerpt"])


def test_search_ignores_non_markdown_files_and_mdx_imports(docs_root: Path) -> None:
    results = docs_search.search("guardrails components")

    assert {result["path"] for result in results} == {"guardrails/index.mdx"}
    assert all("import {" not in str(result["excerpt"]) for result in results)


def test_search_bounds_the_result_count(docs_root: Path) -> None:
    for index in range(15):
        (docs_root / f"provider-{index}.md").write_text("Inference provider setup.\n", encoding="utf-8")
    docs_search._load_index.cache_clear()

    assert len(docs_search.search("inference provider", limit=50)) == docs_search.MAX_LIMIT


def test_search_docs_reports_results_and_empty_states(docs_root: Path) -> None:
    found = json.loads(docs_search.search_docs("inference provider"))
    missing = json.loads(docs_search.search_docs("cheesecake"))
    empty = json.loads(docs_search.search_docs("  "))

    assert found["status"] == "ok"
    assert found["results"][0]["path"] == "inference.md"
    assert missing["status"] == "no_results"
    assert empty["status"] == "error"


def test_search_docs_reports_missing_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docs_search, "docs_dir", lambda: None)

    assert json.loads(docs_search.search_docs("guardrails"))["status"] == "unavailable"


def test_docs_dir_falls_back_to_the_repository_docs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(docs_search.DOCS_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    root = docs_search.docs_dir()

    assert root is not None
    assert (root / "index.mdx").is_file()
