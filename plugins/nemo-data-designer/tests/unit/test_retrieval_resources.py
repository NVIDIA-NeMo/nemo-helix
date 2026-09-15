# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from nemo_data_designer_plugin.retrieval.corpus import _download_fileset, hf_token_from_env, materialize_corpus
from nemo_data_designer_plugin.retrieval.manifest import (
    GENERATION_MANIFEST_SCHEMA_VERSION,
    resolve_generation_input,
    write_generation_manifest,
)
from nemo_data_designer_plugin.retrieval.secrets import resolve_hf_token


def test_fileset_corpus_must_match_job_workspace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match job workspace"):
        _download_fileset(
            "other/documents",
            dest=tmp_path / "corpus",
            sdk=Mock(),
            workspace="default",
        )


def test_local_corpus_path_is_preview_only(tmp_path: Path) -> None:
    corpus = tmp_path / "documents"
    corpus.mkdir()

    assert (
        materialize_corpus(
            str(corpus),
            dest=tmp_path / "unused",
            sdk=Mock(),
            workspace="default",
            allow_local_path=True,
        )
        == corpus.resolve()
    )

    with pytest.raises(ValueError):
        materialize_corpus(
            str(corpus),
            dest=tmp_path / "unused",
            sdk=Mock(),
            workspace="default",
        )


def test_generation_manifest_rejects_non_object_json(tmp_path: Path) -> None:
    manifest = tmp_path / "generation_result.json"
    manifest.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid generation manifest"):
        resolve_generation_input(manifest)


def test_generation_manifest_resolves_relative_output(tmp_path: Path) -> None:
    jsonl = tmp_path / "retrieval_sdg.jsonl"
    jsonl.write_text("{}\n", encoding="utf-8")
    write_generation_manifest(tmp_path, jsonl, dataset_name="retrieval_sdg")
    assert resolve_generation_input(tmp_path / "generation_result.json") == jsonl.resolve()


def test_generation_manifest_rejects_relative_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "generation_result.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
                "dataset_name": "retrieval_sdg",
                "output_path": "../outside.jsonl",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes the manifest directory"):
        resolve_generation_input(manifest)


def test_generation_manifest_rejects_absolute_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "generation_result.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
                "dataset_name": "retrieval_sdg",
                "output_path": str(outside),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes the manifest directory"):
        resolve_generation_input(manifest)


@pytest.mark.parametrize(
    "corpus",
    [
        "hf://org/dataset/../../sibling",
        "hf://org/dataset//sibling",
    ],
)
def test_hf_corpus_rejects_path_traversal(tmp_path: Path, corpus: str) -> None:
    with pytest.raises(ValueError, match="subdirectory"):
        materialize_corpus(
            corpus,
            dest=tmp_path / "corpus",
            sdk=Mock(),
            workspace="default",
        )


def test_hf_corpus_requires_retrieval_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.ModuleType("huggingface_hub"))
    with pytest.raises(ImportError, match=r"nemo-data-designer-plugin\[retrieval-sdg\]"):
        materialize_corpus(
            "hf://org/dataset",
            dest=tmp_path / "corpus",
            sdk=Mock(),
            workspace="default",
        )


def test_generation_and_preview_require_retrieval_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "data_designer_retrieval_sdg", types.ModuleType("data_designer_retrieval_sdg"))
    from nemo_data_designer_plugin.retrieval.generation import execute_generation

    with pytest.raises(ImportError, match=r"Retrieval generate and preview requires"):
        execute_generation(Mock(), preview=True)


def test_hf_corpus_passes_token_to_snapshot_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    fake_hub = types.ModuleType("huggingface_hub")

    def snapshot_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return str(tmp_path / "snapshot")

    setattr(fake_hub, "snapshot_download", snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    materialize_corpus(
        "hf://org/private-dataset",
        dest=tmp_path / "corpus",
        sdk=Mock(),
        workspace="default",
        hf_token="hf_secret_value",
    )

    assert captured["token"] == "hf_secret_value"


def test_hf_token_from_env_reads_step_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert hf_token_from_env() is None
    monkeypatch.setenv("HF_TOKEN", "")
    assert hf_token_from_env() is None
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    assert hf_token_from_env() == "hf_secret_value"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reference", "expected_workspace", "expected_name"),
    [
        ("default/hf-token", "default", "hf-token"),
        ("hf-token", "shared", "hf-token"),
    ],
)
async def test_resolve_hf_token_reads_named_secret(reference: str, expected_workspace: str, expected_name: str) -> None:
    requested: dict[str, str] = {}

    async def access_secret(name: str, workspace: str) -> Mock:
        requested.update(name=name, workspace=workspace)
        return Mock(data=Mock(return_value=Mock(value="hf_secret_value")))

    secrets = Mock(access_secret=access_secret)
    with patch("nemo_data_designer_plugin.retrieval.secrets.client_from_platform", return_value=secrets):
        token = await resolve_hf_token(Mock(), reference, "shared")

    assert token == "hf_secret_value"
    assert requested == {"name": expected_name, "workspace": expected_workspace}


@pytest.mark.asyncio
async def test_resolve_hf_token_without_reference_skips_secrets_service() -> None:
    with patch("nemo_data_designer_plugin.retrieval.secrets.client_from_platform") as client:
        assert await resolve_hf_token(Mock(), None, "default") is None
    client.assert_not_called()
