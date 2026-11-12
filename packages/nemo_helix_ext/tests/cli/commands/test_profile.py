# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from nemo_helix_ext.cli.app import app
from nemo_helix_ext.quickstart.config import QuickstartConfig
from nemo_helix_plugin.auth.profiles.types import Me, PreferenceDocument, PreferenceValue, Profile, ProfileUpdate
from typer.testing import CliRunner, Result

from ..utils import assert_exit_code

runner = CliRunner()

ME = Me(
    account_id="account-123",
    principal_id="alice-sub",
    email="alice@example.com",
    groups=["ml-team"],
    caller_kind="principal",
    auth_method="oidc_bearer",
    profile=Profile(display_name="Alice A.", display_name_override="Alice A.", idp_display_name="Alice Anderson"),
    can_edit_profile=True,
    is_platform_admin=False,
)


def _document(value: dict, effective: dict | None = None, **kwargs) -> PreferenceDocument:
    return PreferenceDocument(
        namespace="studio.ui", value=value, effective=value if effective is None else effective, db_version=3, **kwargs
    )


@pytest.fixture
def profiles() -> Iterator[MagicMock]:
    client = MagicMock()
    client.get_me.return_value.data.return_value = ME
    client.update_my_profile.return_value.data.return_value = ME
    with (
        patch("nemo_helix_ext.quickstart.QuickstartConfig.load", return_value=QuickstartConfig(auth_enabled=False)),
        patch("nemo_helix_ext.cli.core.context.CLIContext.get_client", return_value=MagicMock()),
        patch("nemo_helix_ext.cli.commands.auth.client_from_platform", return_value=client),
        patch("nemo_helix_ext.cli.commands.profile.client_from_platform", return_value=client),
    ):
        yield client


def _invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def _invoke_tty(*args: str) -> Result:
    with patch("nemo_helix_ext.cli.core.api.is_tty", return_value=True):
        return _invoke(*args)


def test_whoami_table(profiles: MagicMock) -> None:
    result = _invoke_tty("auth", "whoami")

    assert_exit_code(result, 0)
    assert "Who am I" in result.output
    for expected in ("account-123", "Alice A.", "Alice Anderson", "alice@example.com", "ml-team", "oidc_bearer"):
        assert expected in result.output


def test_whoami_json(profiles: MagicMock) -> None:
    result = _invoke("--output-format", "json", "auth", "whoami")

    assert_exit_code(result, 0)
    assert Me.model_validate(json.loads(result.output)) == ME


@pytest.mark.parametrize(
    ("args", "body"),
    [
        (["--display-name", "Alice A."], {"display_name": "Alice A."}),
        (["--clear-display-name", "--timezone", "Europe/Paris"], {"display_name": None, "timezone": "Europe/Paris"}),
        (["--clear-locale"], {"locale": None}),
    ],
)
def test_profile_set_sends_only_given_fields(profiles: MagicMock, args: list[str], body: dict) -> None:
    result = _invoke("--output-format", "json", "profile", "set", *args)

    assert_exit_code(result, 0)
    sent: ProfileUpdate = profiles.update_my_profile.call_args.kwargs["body"]
    assert sent.model_dump(exclude_unset=True) == body


@pytest.mark.parametrize("args", [[], ["--display-name", "x", "--clear-display-name"]])
def test_profile_set_rejects_empty_or_conflicting_options(profiles: MagicMock, args: list[str]) -> None:
    result = _invoke("profile", "set", *args)

    assert_exit_code(result, 2)
    profiles.update_my_profile.assert_not_called()


@pytest.fixture
def preferences(profiles: MagicMock) -> Iterator[MagicMock]:
    with patch("nemo_helix_ext.cli.commands.preferences.profiles_client", return_value=profiles):
        yield profiles


def test_get_key_reads_effective_value(preferences: MagicMock) -> None:
    preferences.get_my_preference.return_value.data.return_value = _document(
        {"editor": {"fontSize": 14}}, {"theme": "dark", "editor": {"fontSize": 14}}
    )

    theme = _invoke_tty("preferences", "get", "studio.ui", "--key", "theme")
    font = _invoke("--output-format", "json", "preferences", "get", "studio.ui", "--key", "editor.fontSize")
    missing = _invoke("preferences", "get", "studio.ui", "--key", "theme", "--raw")

    assert_exit_code(theme, 0)
    assert theme.output.strip() == "dark"
    assert font.output.strip() == "14"
    assert_exit_code(missing, 1)
    assert preferences.get_my_preference.call_args.kwargs == {"namespace": "studio.ui", "query_params": None}


def test_get_table_shows_version_and_workspace(preferences: MagicMock) -> None:
    preferences.get_my_preference.return_value.data.return_value = _document({"x": 1}, workspace="default")

    result = _invoke_tty("preferences", "get", "studio.ui", "-w", "default")

    assert_exit_code(result, 0)
    assert "db_version" in result.output and "3" in result.output and "default" in result.output
    assert preferences.get_my_preference.call_args.kwargs["query_params"] == {"workspace": "default"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [("dark", "dark"), ("120", 120), ('{"a": [1]}', {"a": [1]}), ("true", True)],
)
def test_set_key_sends_nested_merge_patch(preferences: MagicMock, value: str, expected: object) -> None:
    preferences.patch_my_preference.return_value.data.return_value = _document({})

    result = _invoke("-f", "json", "preferences", "set", "studio.ui", "--key", "editor.font", value)

    assert_exit_code(result, 0)
    assert preferences.patch_my_preference.call_args.kwargs["body"] == PreferenceValue({"editor": {"font": expected}})


def test_set_json_replaces_document(preferences: MagicMock) -> None:
    preferences.replace_my_preference.return_value.data.return_value = _document({"a": 1})

    result = _invoke("-f", "json", "preferences", "set", "cli", "--json", '{"a": 1}', "-w", "ws")

    assert_exit_code(result, 0)
    kwargs = preferences.replace_my_preference.call_args.kwargs
    assert kwargs == {"namespace": "cli", "body": PreferenceValue({"a": 1}), "query_params": {"workspace": "ws"}}


@pytest.mark.parametrize(
    "args",
    [["--key", "theme"], ["--json", "[1]"], ["--json", "{}", "--key", "a", "b"], []],
)
def test_set_rejects_invalid_combinations(preferences: MagicMock, args: list[str]) -> None:
    assert_exit_code(_invoke("preferences", "set", "cli", *args), 2)


def test_unset_key_patches_null(preferences: MagicMock) -> None:
    preferences.patch_my_preference.return_value.data.return_value = _document({})

    result = _invoke("-f", "json", "preferences", "unset", "studio.ui", "--key", "editor.font")

    assert_exit_code(result, 0)
    assert preferences.patch_my_preference.call_args.kwargs["body"] == PreferenceValue({"editor": {"font": None}})
    preferences.delete_my_preference.assert_not_called()


def test_unset_without_key_deletes_document(preferences: MagicMock) -> None:
    result = _invoke("preferences", "unset", "studio.onboarding", "--workspace", "default")

    assert_exit_code(result, 0)
    preferences.delete_my_preference.assert_called_once_with(
        namespace="studio.onboarding", query_params={"workspace": "default"}
    )
