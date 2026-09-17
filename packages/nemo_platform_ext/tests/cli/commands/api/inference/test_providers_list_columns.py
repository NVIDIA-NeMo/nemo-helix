# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default-table-output coverage for `nemo inference providers list`.

These lock in the provider `status` column: a `LOST` provider must be visually
distinguishable from a `READY` one in the default table output. The default
columns are produced by the CLI generator from `cli_config.yaml`, so this guards
against a regeneration that silently drops or reorders the column.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from nemo_platform_ext.cli.app import app
from nemo_platform_ext.cli.core.context import CLIContext
from typer.testing import CliRunner


def _provider(name: str, status: str) -> dict[str, str]:
    """A minimal provider row carrying the fields the default columns render."""
    return {
        "name": name,
        "status": status,
        "description": f"{name} description",
        "created_at": "2026-01-01T00:00:00",
    }


def _invoke_providers_list() -> str:
    client = MagicMock()
    client.inference.providers.list.return_value = [
        _provider("healthy-provider", "READY"),
        _provider("broken-provider", "LOST"),
    ]
    ctx = CLIContext(
        overrides={"base_url": "http://test.example.com"},
        verbosity=0,
        _client=client,
    )
    result = CliRunner().invoke(app, ["inference", "providers", "list", "--output", "table"], obj=ctx)
    assert result.exit_code == 0, result.output
    return result.output


def test_providers_list_default_table_shows_status_column() -> None:
    output = _invoke_providers_list()
    # The status column header renders from the field name.
    assert "status" in output
    # Both providers are shown (non-lossy) and their statuses are visible.
    assert "READY" in output
    assert "LOST" in output
    assert "healthy-provider" in output
    assert "broken-provider" in output


def test_providers_list_status_column_precedes_description() -> None:
    output = _invoke_providers_list()
    # Column order: name, status, description, created_at — status near the front
    # (before description) so it's scannable.
    assert output.index("status") < output.index("description")
