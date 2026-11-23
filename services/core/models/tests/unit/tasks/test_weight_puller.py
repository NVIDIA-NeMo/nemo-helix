# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the packaged weight puller's remote-access and startup contracts."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import respx
from nhx.core.models.tasks import weight_puller


def test_download_pins_remote_commit_and_reuses_cached_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cached = tmp_path / "cached-blob"
    cached.write_bytes(b"model weights")
    api = MagicMock()
    api.model_info.return_value = SimpleNamespace(
        sha="a" * 40, siblings=[SimpleNamespace(rfilename="weights/model.safetensors")]
    )
    api_factory = MagicMock(return_value=api)
    download = MagicMock(return_value=str(cached))
    monkeypatch.setattr(weight_puller, "HfApi", api_factory)
    monkeypatch.setattr(weight_puller, "hf_hub_download", download)
    monkeypatch.setenv("HF_ENDPOINT", "http://proxy.test/apis/files/v2/hf")
    monkeypatch.setenv("HF_TOKEN", "placeholder")

    target = tmp_path / "model"
    weight_puller.pull_weights("workspace/model", target, revision="main")

    api_factory.assert_called_once_with(endpoint="http://proxy.test/apis/files/v2/hf", token="placeholder")
    api.model_info.assert_called_once_with("workspace/model", revision="main")
    download.assert_called_once_with(
        "workspace/model",
        "weights/model.safetensors",
        revision="a" * 40,
        endpoint="http://proxy.test/apis/files/v2/hf",
        token="placeholder",
    )
    materialized = target / "weights/model.safetensors"
    assert materialized.read_bytes() == b"model weights"
    assert materialized.stat().st_ino == cached.stat().st_ino


@pytest.mark.parametrize("deny_manifest", [False, True])
def test_remote_failure_does_not_report_stale_weights_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deny_manifest: bool
) -> None:
    target = tmp_path / "model"
    target.mkdir()
    stale = target / "config.json"
    stale.write_text("stale weights")
    (target / ".cache").mkdir()
    api = MagicMock()
    api.model_info.return_value = SimpleNamespace(sha="a" * 40, siblings=[SimpleNamespace(rfilename="config.json")])
    error = httpx.HTTPStatusError(
        "Access denied", request=httpx.Request("GET", "http://files.test"), response=httpx.Response(403)
    )
    if deny_manifest:
        api.model_info.side_effect = error
    download = MagicMock(side_effect=error)
    monkeypatch.setattr(weight_puller, "HfApi", MagicMock(return_value=api))
    monkeypatch.setattr(weight_puller, "hf_hub_download", download)

    assert weight_puller.main(["--repo-id", "workspace/model", "--local-dir", str(target)]) == 1
    assert stale.read_text() == "stale weights"
    if deny_manifest:
        download.assert_not_called()


def test_puller_waits_before_downloading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wait = MagicMock()
    pull = MagicMock()
    calls = MagicMock()
    calls.attach_mock(wait, "wait")
    calls.attach_mock(pull, "pull")
    monkeypatch.setattr(weight_puller, "wait_for_proxy", wait)
    monkeypatch.setattr(weight_puller, "pull_weights", pull)

    assert (
        weight_puller.main(
            [
                "--repo-id",
                "workspace/model",
                "--local-dir",
                str(tmp_path),
                "--revision",
                "main",
                "--proxy-health-url",
                "http://127.0.0.1:8090/healthz",
            ]
        )
        == 0
    )
    assert [call[0] for call in calls.mock_calls] == ["wait", "pull"]
    pull.assert_called_once_with("workspace/model", tmp_path, revision="main")


@respx.mock
def test_proxy_startup_retries_until_ready() -> None:
    route = respx.get("http://proxy.test/healthz").mock(side_effect=[httpx.Response(503), httpx.Response(200)])
    weight_puller.wait_for_proxy("http://proxy.test/healthz", timeout=5)
    assert route.call_count == 2


@respx.mock
def test_proxy_startup_has_a_deadline() -> None:
    respx.get("http://proxy.test/healthz").mock(return_value=httpx.Response(503))
    with pytest.raises(RuntimeError, match="did not become ready"):
        weight_puller.wait_for_proxy("http://proxy.test/healthz", timeout=0)


def test_proxy_startup_failure_stops_the_puller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(weight_puller, "wait_for_proxy", MagicMock(side_effect=RuntimeError("not ready")))
    download = MagicMock()
    monkeypatch.setattr(weight_puller, "pull_weights", download)
    assert (
        weight_puller.main(
            ["--repo-id", "workspace/model", "--local-dir", str(tmp_path), "--proxy-health-url", "http://proxy/healthz"]
        )
        == 1
    )
    download.assert_not_called()
