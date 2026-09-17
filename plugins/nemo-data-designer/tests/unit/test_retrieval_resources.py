# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest
from data_designer_nemo.errors import NDDInternalError, NDDInvalidConfigError
from nemo_data_designer_plugin.retrieval.corpus import _download_fileset, hf_token_from_env, materialize_corpus
from nemo_data_designer_plugin.retrieval.manifest import (
    GENERATION_MANIFEST_SCHEMA_VERSION,
    resolve_generation_input,
    write_generation_manifest,
)
from nemo_data_designer_plugin.retrieval.secrets import resolve_hf_token
from nemo_platform_plugin.client.errors import InternalServerError, NotFoundError, PermissionDeniedError


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


def test_hf_file_uri_includes_filename_in_allow_patterns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    fake_hub = types.ModuleType("huggingface_hub")
    dest = tmp_path / "snapshot"
    dest.mkdir()
    (dest / "nv_pp_dd_sdg.json").write_text("[]\n", encoding="utf-8")

    def snapshot_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return str(dest)

    setattr(fake_hub, "snapshot_download", snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    staged = materialize_corpus(
        "hf://org/dataset@abc123/nv_pp_dd_sdg.json",
        dest=dest,
        sdk=Mock(),
        workspace="default",
    )

    assert captured["allow_patterns"] == ["nv_pp_dd_sdg.json", "nv_pp_dd_sdg.json/**"]
    assert staged == dest / "nv_pp_dd_sdg.json"


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
async def test_resolve_hf_token_rejects_malformed_secret_reference() -> None:
    with patch("nemo_data_designer_plugin.retrieval.secrets.client_from_platform") as client:
        with pytest.raises(NDDInvalidConfigError, match="formatted incorrectly"):
            await resolve_hf_token(Mock(), "too/many/parts", "default")

    client.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            NotFoundError(httpx.Response(404, json={"detail": "missing"}, request=httpx.Request("POST", "http://x"))),
            "Could not find secret 'hf-token' in workspace 'default'",
        ),
        (
            PermissionDeniedError(
                httpx.Response(403, json={"detail": "denied"}, request=httpx.Request("POST", "http://x"))
            ),
            "Access denied to workspace 'default'",
        ),
    ],
)
async def test_resolve_hf_token_maps_missing_or_denied_secret_to_invalid_config(error: Exception, message: str) -> None:
    async def access_secret(name: str, workspace: str) -> Mock:
        del name, workspace
        raise error

    secrets = Mock(access_secret=access_secret)
    with patch("nemo_data_designer_plugin.retrieval.secrets.client_from_platform", return_value=secrets):
        with pytest.raises(NDDInvalidConfigError, match=message):
            await resolve_hf_token(Mock(), "default/hf-token", "shared")


@pytest.mark.asyncio
async def test_resolve_hf_token_maps_unexpected_secret_error_to_internal_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def access_secret(name: str, workspace: str) -> Mock:
        del name, workspace
        raise InternalServerError(
            httpx.Response(500, json={"detail": "boom"}, request=httpx.Request("POST", "http://x"))
        )

    secrets = Mock(access_secret=access_secret)
    with patch("nemo_data_designer_plugin.retrieval.secrets.client_from_platform", return_value=secrets):
        with caplog.at_level(logging.ERROR, logger="nemo_data_designer_plugin.retrieval.secrets"):
            with pytest.raises(NDDInternalError, match="unexpected error occurred"):
                await resolve_hf_token(Mock(), "default/hf-token", "shared")

    assert "hf-token" not in caplog.text
    assert "default/hf-token" not in caplog.text


@pytest.mark.asyncio
async def test_resolve_hf_token_without_reference_skips_secrets_service() -> None:
    with patch("nemo_data_designer_plugin.retrieval.secrets.client_from_platform") as client:
        assert await resolve_hf_token(Mock(), None, "default") is None
    client.assert_not_called()
