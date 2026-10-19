# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared mechanics for optimizer.yaml without a universal profile schema."""

import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

PROFILE_FILENAME = "optimizer.yaml"
DEFAULT_BASE_URL = "http://localhost:8080"
_ETHOS_FILENAMES = ("ETHOS.md", "README.md")

ProfileModel = TypeVar("ProfileModel", bound=BaseModel)


class ProfileError(ValueError):
    """A profile file or profile-owned path is invalid."""


class EnvFileError(ValueError):
    """An adjacent environment file could not be read safely."""


def discover_profile(start: Path | None = None) -> Path | None:
    """Walk from *start* or cwd to the filesystem root for optimizer.yaml."""
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / PROFILE_FILENAME
        if candidate.is_file():
            return candidate
    return None


def resolve_profile_path(value: str, profile_dir: Path) -> Path:
    """Resolve an absolute, home-relative, or profile-relative path."""
    try:
        path = Path(value).expanduser()
    except RuntimeError as exc:
        raise ProfileError(f"Could not resolve path {value!r}: {exc}") from None
    return path.resolve() if path.is_absolute() else (profile_dir / path).resolve()


def load_profile_model(path: Path, model: type[ProfileModel]) -> ProfileModel:
    """Load optimizer.yaml into a caller-owned strict or tolerant model."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeError as exc:
        raise ProfileError(f"Could not parse profile {path}: expected readable UTF-8 YAML ({exc})") from None
    except (OSError, yaml.YAMLError) as exc:
        raise ProfileError(f"Could not parse profile {path}: {exc}") from None
    if not isinstance(payload, dict):
        raise ProfileError(f"Could not parse profile {path}: expected a YAML mapping")
    if "profile_dir" in payload:
        raise ProfileError(f"Invalid profile {path}: 'profile_dir' is reserved")
    values = dict(payload)
    values["profile_dir"] = path.parent.resolve()
    try:
        return model.model_validate(values)
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(str(item) for item in error['loc'])}: {error['msg']}" for error in exc.errors())
        raise ProfileError(f"Invalid profile {path}: {details}") from None


def load_env_file(path: Path, env: MutableMapping[str, str] = os.environ) -> list[str]:
    """Load simple KEY=VALUE entries without replacing existing environment keys."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    except (OSError, UnicodeError) as exc:
        raise EnvFileError(
            f"Could not read environment file {path}: {exc}. Check that the file is readable UTF-8 text, then retry."
        ) from None
    loaded: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and key not in env:
            env[key] = value
            loaded.append(key)
    return loaded


def resolve_ethos_path(profile_dir: Path, configured: str | None) -> Path | None:
    """Resolve a configured Ethos or the conventional profile-local file."""
    if configured is not None:
        path = resolve_profile_path(configured, profile_dir)
        if not path.is_file():
            raise ProfileError(f"Profile ethos {configured!r} does not exist (resolved to {path})")
        return path
    for name in _ETHOS_FILENAMES:
        candidate = profile_dir / name
        if candidate.is_file():
            return candidate
    return None


def resolve_base_url(explicit: str | None, env: Mapping[str, str] = os.environ) -> str:
    """Apply explicit, NHX_BASE_URL, then localhost precedence."""
    if explicit is not None:
        return explicit
    env_url = env.get("NHX_BASE_URL")
    return env_url if env_url is not None else DEFAULT_BASE_URL
