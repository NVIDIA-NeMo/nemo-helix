# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for customizer E2E asset helpers (no platform required)."""

import ast
import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import httpx
import pytest

from e2e.customizer import mine_embedding_data, stage_assets
from e2e.customizer.assets_manifest import load, local_only_formats
from e2e.customizer.generate_grpo_assets import (
    ADAPTER_ENV_FORMAT,
    ADAPTER_SMOKE_FORMAT,
    ENV_FORMAT,
    NATIVE_ENV_FORMAT,
    NATIVE_REF_ENV_FORMAT,
    _bake_default,
)
from e2e.customizer.mine_embedding_data import _download_resumable, _write_subsampled, _write_training_sample

_PAYLOAD = bytes(range(256)) * 40  # 10,240 bytes


class _DroppingStream(httpx.SyncByteStream):
    """Yields part of a body, then fails the way a dropped port-forward does."""

    def __init__(self, data: bytes, drop_after: int) -> None:
        self._data, self._drop_after = data, drop_after

    def __iter__(self) -> Iterator[bytes]:
        yield self._data[: self._drop_after]
        raise httpx.RemoteProtocolError("Server disconnected without sending a response.")


def _ranged_server(*, drop_first_chunk_after: int | None = None, honor_range: bool = True):
    """Serve ``_PAYLOAD`` like the Files API: a range past the end gets 416, not a shorter body."""
    requests: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request.headers["Range"])
        if not honor_range:
            return httpx.Response(200, content=_PAYLOAD)
        start, end = (int(part) for part in request.headers["Range"].removeprefix("bytes=").split("-"))
        if end >= len(_PAYLOAD):
            return httpx.Response(416, headers={"Content-Range": f"bytes */{len(_PAYLOAD)}"})
        body = _PAYLOAD[start : end + 1]
        headers = {"Content-Range": f"bytes {start}-{end}/{len(_PAYLOAD)}"}
        if drop_first_chunk_after is not None and len(requests) == 1:
            return httpx.Response(206, headers=headers, stream=_DroppingStream(body, drop_first_chunk_after))
        return httpx.Response(206, headers=headers, content=body)

    return httpx.MockTransport(handle), requests


@pytest.fixture
def no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mine_embedding_data.time, "sleep", lambda _seconds: None)


def test_download_resumable_keeps_ranges_inside_the_file(tmp_path: Path) -> None:
    transport, requests = _ranged_server()
    dest = tmp_path / "queries.jsonl"

    _download_resumable("http://files/x", dest, len(_PAYLOAD), headers={}, chunk_bytes=4096, transport=transport)

    assert dest.read_bytes() == _PAYLOAD
    assert requests == ["bytes=0-4095", "bytes=4096-8191", "bytes=8192-10239"]


def test_download_resumable_downloads_a_file_smaller_than_a_chunk(tmp_path: Path) -> None:
    transport, requests = _ranged_server()
    dest = tmp_path / "test.tsv"

    _download_resumable("http://files/x", dest, len(_PAYLOAD), headers={}, transport=transport)

    assert dest.read_bytes() == _PAYLOAD
    assert requests == [f"bytes=0-{len(_PAYLOAD) - 1}"]


def test_download_resumable_resumes_a_dropped_range(tmp_path: Path, no_retry_sleep: None) -> None:
    transport, requests = _ranged_server(drop_first_chunk_after=1000)
    dest = tmp_path / "training.jsonl"

    _download_resumable("http://files/x", dest, len(_PAYLOAD), headers={}, chunk_bytes=4096, transport=transport)

    assert dest.read_bytes() == _PAYLOAD
    assert requests[:2] == ["bytes=0-4095", "bytes=1000-5095"]


def test_download_resumable_continues_a_partial_file(tmp_path: Path) -> None:
    transport, requests = _ranged_server()
    dest = tmp_path / "training.jsonl"
    dest.write_bytes(_PAYLOAD[:3000])

    _download_resumable("http://files/x", dest, len(_PAYLOAD), headers={}, chunk_bytes=4096, transport=transport)

    assert dest.read_bytes() == _PAYLOAD
    assert requests[0] == "bytes=3000-7095"


def test_download_resumable_restarts_when_range_is_ignored(tmp_path: Path) -> None:
    transport, _ = _ranged_server(honor_range=False)
    dest = tmp_path / "training.jsonl"
    dest.write_bytes(b"stale partial download")

    _download_resumable("http://files/x", dest, len(_PAYLOAD), headers={}, chunk_bytes=4096, transport=transport)

    assert dest.read_bytes() == _PAYLOAD


def test_download_resumable_fails_instead_of_stopping_short(tmp_path: Path, no_retry_sleep: None) -> None:
    def always_unsatisfiable(request: httpx.Request) -> httpx.Response:
        return httpx.Response(416)

    transport = httpx.MockTransport(always_unsatisfiable)
    with pytest.raises(SystemExit, match="Gave up"):
        _download_resumable("http://files/x", tmp_path / "x", len(_PAYLOAD), headers={}, transport=transport)


def test_write_subsampled_fails_without_eval_queries(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_beir(src)
    (src / "eval_beir" / "qrels" / "test.tsv").write_text("\n")

    with pytest.raises(SystemExit, match="No eval queries"):
        _write_subsampled(src, tmp_path / "dst", max_train_rows=None, max_eval_queries=None, seed=42)


def _unpublished_manifest(output_dir: Path) -> dict:
    """The real manifest with the NVDocs dataset marked unpublished and read from ``output_dir``."""
    manifest = load()
    manifest["generation"]["output_dir"] = str(output_dir)
    manifest["datasets"]["nvdocs_retrieval"]["s3_published"] = False
    return manifest


def test_fixtures_only_stage_assets_the_manifest_publishes() -> None:
    """Every S3 prefix the fixtures sync is one ``publish_assets_to_s3.sh`` uploads."""
    tree = ast.parse((Path(__file__).parent / "conftest.py").read_text())
    synced: dict[str, set[str]] = {"sync_dataset_format": set(), "sync_model": set()}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in synced:
            prefix = node.args[0]
            assert isinstance(prefix, ast.Constant) and isinstance(prefix.value, str)
            synced[node.func.attr].add(prefix.value)
    manifest = load()

    published_formats = {name for dataset in manifest["datasets"].values() for name in dataset["outputs"]}
    published_models = {model["s3_folder"] for model in manifest["models"]}
    assert synced["sync_dataset_format"] and synced["sync_dataset_format"] <= published_formats
    assert synced["sync_model"] and synced["sync_model"] <= published_models


def test_prefetch_covers_every_fixture_that_syncs_from_s3() -> None:
    tree = ast.parse((Path(__file__).parent / "conftest.py").read_text())
    syncing = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in {"sync_dataset_format", "sync_model"}
            for call in ast.walk(node)
        )
    }
    prefetched = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "S3_ASSET_FIXTURES" for t in node.targets)
    )

    assert syncing == set(prefetched)


def test_fixtures_to_prefetch_skips_tests_that_will_not_run() -> None:
    class _Item:
        def __init__(self, fixturenames: list[str], skipped: bool = False) -> None:
            self.fixturenames = fixturenames
            self._skipped = skipped

        def get_closest_marker(self, name: str) -> object | None:
            return object() if name == "skip" and self._skipped else None

    items = [
        _Item(["client", "customizer_asset_cache", "embed_model_cache"]),
        _Item(["grpo_math_env_cache", "customizer_asset_cache"], skipped=True),
        _Item(["client"]),
    ]
    assets = ("customizer_asset_cache", "embed_model_cache", "grpo_math_env_cache")

    assert stage_assets.fixtures_to_prefetch(cast(list[pytest.Item], items), assets) == {
        "customizer_asset_cache",
        "embed_model_cache",
    }


def test_local_only_formats_lists_unpublished_datasets(tmp_path: Path) -> None:
    assert local_only_formats(_unpublished_manifest(tmp_path)) == {
        "embedding_nvdocs",
        "embedding_nvdocs_smoke",
        *GRPO_BUILT_FORMATS,
    }


# CI builds these from the platform checkout so they match the image's NeMo-RL pin.
GRPO_BUILT_FORMATS = {
    "grpo_math_env",
    "grpo_math_env_native",
    "grpo_math_env_native_ref",
    "grpo_ascii_tree_env",
    "grpo_ascii_tree_smoke",
}


def test_grpo_envs_are_built_locally_not_synced_from_s3() -> None:
    assert GRPO_BUILT_FORMATS <= local_only_formats(load())


def test_generator_writes_every_built_grpo_format() -> None:
    written = {ENV_FORMAT, NATIVE_ENV_FORMAT, NATIVE_REF_ENV_FORMAT, ADAPTER_ENV_FORMAT, ADAPTER_SMOKE_FORMAT}
    assert written == GRPO_BUILT_FORMATS


def test_missing_built_env_names_its_generator(tmp_path: Path) -> None:
    with pytest.raises(pytest.fail.Exception, match=r"generate_grpo_assets\.py --env-only"):
        stage_assets.sync_dataset_format("grpo_math_env", manifest=_unpublished_manifest(tmp_path))


def test_sync_dataset_format_reads_unpublished_dataset_locally(tmp_path: Path) -> None:
    local_dir = tmp_path / "embedding_nvdocs"
    local_dir.mkdir()
    (local_dir / "training.jsonl").write_text("{}\n")

    assert stage_assets.sync_dataset_format("embedding_nvdocs", manifest=_unpublished_manifest(tmp_path)) == local_dir


def test_sync_dataset_format_fails_when_unpublished_dataset_missing(tmp_path: Path) -> None:
    with pytest.raises(pytest.fail.Exception, match="not published to S3 yet"):
        stage_assets.sync_dataset_format("embedding_nvdocs", manifest=_unpublished_manifest(tmp_path))


def test_head_jsonl_keeps_first_non_empty_rows(tmp_path: Path) -> None:
    source = tmp_path / "training.jsonl"
    source.write_text('{"i": 0}\n\n{"i": 1}\n{"i": 2}\n{"i": 3}')

    dest = stage_assets._head_jsonl(source, tmp_path / "out" / "training.jsonl", max_rows=2)

    assert dest.read_text() == '{"i": 0}\n{"i": 1}\n'


def test_bake_default_reads_variable_default() -> None:
    bake = (
        'variable "NEMO_RL_REPO" {\n  default = "https://github.com/example/RL.git"\n}\n'
        'variable "NEMO_RL_REF" {\n  default = "abc123" # pinned commit\n}\n'
    )

    assert _bake_default(bake, "NEMO_RL_REPO") == "https://github.com/example/RL.git"
    assert _bake_default(bake, "NEMO_RL_REF") == "abc123"
    with pytest.raises(SystemExit, match="NEMO_GYM_REF"):
        _bake_default(bake, "NEMO_GYM_REF")


def test_write_training_sample_caps_rows(tmp_path: Path) -> None:
    source = tmp_path / "training.jsonl"
    source.write_text("".join(json.dumps({"query": f"q{i}"}) + "\n" for i in range(10)))

    _write_training_sample(source, tmp_path / "smoke", rows=4, seed=42)

    rows = (tmp_path / "smoke" / "training.jsonl").read_text().splitlines()
    assert len(rows) == 4
    assert set(rows) <= set(source.read_text().splitlines())
    assert not (tmp_path / "smoke" / "eval_beir").exists()


def _write_beir(root: Path) -> None:
    beir = root / "eval_beir"
    (beir / "qrels").mkdir(parents=True)
    (root / "training.jsonl").write_text("".join(json.dumps({"query": f"q{i}"}) + "\n" for i in range(10)))
    (beir / "corpus.jsonl").write_text("".join(json.dumps({"_id": f"d{i}", "text": "t"}) + "\n" for i in range(4)))
    (beir / "queries.jsonl").write_text("".join(json.dumps({"_id": f"q{i}", "text": "t"}) + "\n" for i in range(5)))
    (beir / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        "q0\td0\t1\n"
        "q0\td0\t1\n"  # duplicate judgment the BEIR loader rejects
        "q1\td1\t1\n"
        "q2\td2\t1\n"
        "q3\td3\t1\n"
        "q4\td0\t1\n"
    )


def test_write_subsampled_caps_rows_and_dedupes_qrels(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_beir(src)

    _write_subsampled(src, dst, max_train_rows=3, max_eval_queries=2, seed=42)

    assert len((dst / "training.jsonl").read_text().splitlines()) == 3

    qrels = (dst / "eval_beir" / "qrels" / "test.tsv").read_text().splitlines()
    assert qrels[0] == "query-id\tcorpus-id\tscore"
    judgments = [tuple(line.split("\t")[:2]) for line in qrels[1:]]
    assert len(judgments) == len(set(judgments))
    kept_queries = {query_id for query_id, _ in judgments}
    assert len(kept_queries) == 2

    queries = [json.loads(line)["_id"] for line in (dst / "eval_beir" / "queries.jsonl").read_text().splitlines()]
    assert set(queries) == kept_queries
    assert (dst / "eval_beir" / "corpus.jsonl").read_text() == (src / "eval_beir" / "corpus.jsonl").read_text()


def test_write_subsampled_keeps_everything_under_caps(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_beir(src)

    _write_subsampled(src, dst, max_train_rows=100, max_eval_queries=100, seed=42)

    assert len((dst / "training.jsonl").read_text().splitlines()) == 10
    assert len((dst / "eval_beir" / "qrels" / "test.tsv").read_text().splitlines()) == 1 + 5
    assert len((dst / "eval_beir" / "queries.jsonl").read_text().splitlines()) == 5


def test_write_subsampled_keeps_everything_without_caps(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_beir(src)

    _write_subsampled(src, dst, max_train_rows=None, max_eval_queries=None, seed=42)

    assert len((dst / "training.jsonl").read_text().splitlines()) == 10
    # Still unique-keyed: the duplicate q0/d0 judgment is dropped.
    assert len((dst / "eval_beir" / "qrels" / "test.tsv").read_text().splitlines()) == 1 + 5
    assert len((dst / "eval_beir" / "queries.jsonl").read_text().splitlines()) == 5
