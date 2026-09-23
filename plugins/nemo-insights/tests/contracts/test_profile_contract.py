# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from nemo_insights_plugin.contracts.profile import (
    DEFAULT_BASE_URL,
    EnvFileError,
    ProfileError,
    discover_profile,
    load_env_file,
    load_profile_model,
    resolve_base_url,
    resolve_ethos_path,
    resolve_profile_path,
)
from pydantic import BaseModel, ConfigDict


class TolerantProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    agent: str
    profile_dir: Path


class StrictProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str
    profile_dir: Path


def test_load_profile_model_respects_caller_strictness_and_injects_directory(tmp_path: Path) -> None:
    path = tmp_path / "optimizer.yaml"
    path.write_text("agent: a\nexperiment_only: true\n", encoding="utf-8")

    assert load_profile_model(path, TolerantProfile).profile_dir == tmp_path.resolve()
    with pytest.raises(ProfileError, match="experiment_only"):
        load_profile_model(path, StrictProfile)


def test_load_profile_model_rejects_reserved_directory(tmp_path: Path) -> None:
    path = tmp_path / "optimizer.yaml"
    path.write_text("agent: a\nprofile_dir: /tmp/forged\n", encoding="utf-8")

    with pytest.raises(ProfileError, match="reserved"):
        load_profile_model(path, TolerantProfile)


def test_load_profile_model_wraps_yaml_utf8_and_shape_errors(tmp_path: Path) -> None:
    path = tmp_path / "optimizer.yaml"
    path.write_bytes(b"agent: \xff")
    with pytest.raises(ProfileError, match="UTF-8"):
        load_profile_model(path, TolerantProfile)

    path.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="YAML mapping"):
        load_profile_model(path, TolerantProfile)


def test_discover_profile_walks_up_and_returns_none_when_absent(tmp_path: Path) -> None:
    child = tmp_path / "one" / "two"
    child.mkdir(parents=True)
    assert discover_profile(child) is None

    profile = tmp_path / "optimizer.yaml"
    profile.write_text("agent: a\n", encoding="utf-8")
    assert discover_profile(child) == profile


def test_resolve_profile_path_handles_relative_absolute_and_home(tmp_path: Path) -> None:
    relative = resolve_profile_path("./agent", tmp_path)
    absolute = resolve_profile_path(str(tmp_path / "agent"), Path("/elsewhere"))

    assert relative == (tmp_path / "agent").resolve()
    assert absolute == (tmp_path / "agent").resolve()


def test_resolve_profile_path_wraps_unresolvable_home_as_profile_error(tmp_path: Path) -> None:
    with pytest.raises(ProfileError, match="no-such-user-anywhere"):
        resolve_profile_path("~no-such-user-anywhere/agent", tmp_path)


def test_load_env_file_parses_without_overriding(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        '# comment\nPLAIN=value\nexport EXPORTED=ok\nQUOTED="with spaces"\nSET=file\n',
        encoding="utf-8",
    )
    env = {"SET": "process"}

    assert load_env_file(path, env) == ["PLAIN", "EXPORTED", "QUOTED"]
    assert env == {"SET": "process", "PLAIN": "value", "EXPORTED": "ok", "QUOTED": "with spaces"}
    assert load_env_file(tmp_path / "missing.env", env) == []


def test_load_env_file_strips_exactly_one_matched_quote_pair(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "SINGLE='quoted'\nAPOSTROPHE=\"it's\"\nMISMATCHED=\"value'\nTRAILING=value'\nNESTED=\"\"twice\"\"\nEMPTY=''\n",
        encoding="utf-8",
    )
    env: dict[str, str] = {}

    load_env_file(path, env)

    assert env == {
        "SINGLE": "quoted",
        "APOSTROPHE": "it's",
        "MISMATCHED": "\"value'",
        "TRAILING": "value'",
        "NESTED": '"twice"',
        "EMPTY": "",
    }


def test_load_env_file_wraps_read_failures_without_raw_chain(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_bytes(b"KEY=\xff")

    with pytest.raises(EnvFileError, match="readable UTF-8") as exc_info:
        load_env_file(path, {})

    assert exc_info.value.__cause__ is None


def test_load_env_file_wraps_permission_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / ".env"
    path.write_text("KEY=value\n", encoding="utf-8")
    original_read_text = Path.read_text

    def deny(candidate: Path, *args: object, **kwargs: object) -> str:
        if candidate == path:
            raise PermissionError("permission denied")
        return original_read_text(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny)

    with pytest.raises(EnvFileError, match="permission denied") as exc_info:
        load_env_file(path, {})

    assert exc_info.value.__cause__ is None


def test_resolve_ethos_uses_configured_then_conventional_precedence(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Readme", encoding="utf-8")
    assert resolve_ethos_path(tmp_path, None) == readme

    ethos = tmp_path / "ETHOS.md"
    ethos.write_text("# Ethos", encoding="utf-8")
    assert resolve_ethos_path(tmp_path, None) == ethos
    assert resolve_ethos_path(tmp_path, "./README.md") == readme.resolve()

    with pytest.raises(ProfileError, match="does not exist"):
        resolve_ethos_path(tmp_path, "./missing.md")


def test_resolve_base_url_uses_only_explicit_nhx_and_default() -> None:
    env = {"NHX_BASE_URL": "http://nhx", "NEMO_BASE_URL": "http://ignored"}

    assert resolve_base_url("http://flag", env) == "http://flag"
    assert resolve_base_url(None, env) == "http://nhx"
    assert resolve_base_url(None, {"NEMO_BASE_URL": "http://ignored"}) == DEFAULT_BASE_URL
    assert resolve_base_url("", env) == ""  # explicit empty string is not silently replaced
