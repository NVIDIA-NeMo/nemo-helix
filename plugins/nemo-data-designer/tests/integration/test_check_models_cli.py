# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ``nemo data-designer check-models``.

The CLI command is a thin shell over
:func:`nemo_data_designer_plugin.sdk.check_models.check_models_config`; these
tests assert the public CLI behavior (stdout / exit code / JSON shape) against
an in-process mock platform.

The probe itself is patched throughout. As
``test_preview_remote_sdk.test_preview_model_health_check_failure_returns_validation_error``
documents, a real health check cannot succeed here: the request is issued by
the Data Designer library's own HTTP client, which does not carry the ASGI
transport the tests configure. What these tests cover is everything the plugin
owns around that call — resolution, error shaping, log forwarding, exit codes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import nemo_data_designer_plugin.testing.utils as u
import pytest
from data_designer.engine.models.errors import ModelNotFoundError
from data_designer.interface.data_designer import DataDesigner

pytestmark = pytest.mark.integration

# Shape mirrors the real engine log that names the alias being probed; the
# error the engine raises on failure does not carry it.
_PROBE_LOG = "👀 Checking 'nano-v3' in provider named 'p' for model alias 'text'..."


def _write_valid_config(tmp_path: Path) -> Path:
    return u.write_config_file(
        tmp_path,
        f"""
import data_designer.config as dd


def load_config_builder() -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="text", model={u.ENABLED_MODEL_NAME!r},
                provider={u.OPEN_PROVIDER_NAME!r},
            )
        ]
    )
    builder.add_column(
        dd.LLMTextColumnConfig(name="x", prompt="hi", model_alias="text")
    )
    return builder
""",
        name="valid_config.py",
    )


def _write_unresolvable_provider_config(tmp_path: Path) -> Path:
    return u.write_config_file(
        tmp_path,
        f"""
import data_designer.config as dd


def load_config_builder() -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="text", model={u.ENABLED_MODEL_NAME!r},
                provider="default/does-not-exist",
            )
        ]
    )
    builder.add_column(
        dd.LLMTextColumnConfig(name="x", prompt="hi", model_alias="text")
    )
    return builder
""",
        name="unresolvable_provider_config.py",
    )


def _patch_probe(monkeypatch: pytest.MonkeyPatch, outcome: Exception | None, *, emit_log: bool = False) -> list[bool]:
    """Patch the engine probe, recording whether it was reached."""
    calls: list[bool] = []

    def _probe(*args, **kwargs) -> None:
        calls.append(True)
        if emit_log:
            logging.getLogger("data_designer.engine.models.registry").info(_PROBE_LOG)
        if outcome is not None:
            raise outcome

    monkeypatch.setattr(DataDesigner, "check_models", _probe)
    return calls


def test_check_models_reports_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_valid_config(tmp_path)
    calls = _patch_probe(monkeypatch, None)

    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_providers(client_context),
    ):
        result = u.invoke_cli(["check-models", str(config_path)], client_context)

    assert result.exit_code == 0, result.output
    assert calls == [True]
    assert "All models responded successfully" in result.output


def test_check_models_reports_failing_model_with_error_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_valid_config(tmp_path)
    _patch_probe(monkeypatch, ModelNotFoundError("model 'nano-v3' does not exist"))

    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_providers(client_context),
    ):
        result = u.invoke_cli(["check-models", str(config_path)], client_context)

    assert result.exit_code == 1, result.output
    # The rendered report hard-wraps to the console width, so compare against
    # whitespace-normalized output rather than chasing line breaks.
    output = " ".join(result.output.split())
    # The error type is what distinguishes a missing model from an auth or
    # connectivity failure, so it has to survive into the rendered output.
    assert "ModelNotFoundError" in output
    assert "model 'nano-v3' does not exist" in output


def test_check_models_forwards_engine_logs_naming_the_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The alias being probed appears only in the engine's logs, so a fail-fast
    probe is only diagnosable if those logs reach the user."""
    config_path = _write_valid_config(tmp_path)
    _patch_probe(monkeypatch, ModelNotFoundError("nope"), emit_log=True)

    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_providers(client_context),
    ):
        result = u.invoke_cli(["check-models", str(config_path)], client_context)

    assert result.exit_code == 1, result.output
    assert "model alias 'text'" in " ".join(result.output.split())


def test_check_models_skips_probe_when_config_cannot_resolve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unresolvable provider is reported here rather than deferred to
    ``validate``, and the engine is never asked to probe it."""
    config_path = _write_unresolvable_provider_config(tmp_path)
    calls = _patch_probe(monkeypatch, None)

    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_providers(client_context),
    ):
        result = u.invoke_cli(["check-models", str(config_path)], client_context)

    assert result.exit_code == 1, result.output
    assert calls == []
    assert "does-not-exist" in " ".join(result.output.split())


def test_check_models_json_output_is_machine_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_valid_config(tmp_path)
    _patch_probe(monkeypatch, ModelNotFoundError("nope"), emit_log=True)

    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_providers(client_context),
    ):
        result = u.invoke_cli(["check-models", str(config_path), "--output", "json"], client_context)

    assert result.exit_code == 1
    # Engine logs are suppressed for json so the document is the only thing on
    # stdout and stays parseable as a whole.
    payload = json.loads(result.output.strip())
    assert payload["ok"] is False
    assert payload["config_source"] == str(config_path)
    assert payload["errors"] == [{"error_type": "ModelNotFoundError", "message": "nope"}]


def test_check_models_json_output_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_valid_config(tmp_path)
    _patch_probe(monkeypatch, None)

    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_providers(client_context),
    ):
        result = u.invoke_cli(["check-models", str(config_path), "--output", "json"], client_context)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert payload["ok"] is True
    assert payload["errors"] == []
