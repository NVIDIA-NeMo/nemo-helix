# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration manager for NemoClient.

Reads ``~/.config/nmp/config.yaml`` (the user-facing CLI/SDK config) and
resolves clusters, users, and contexts into a :class:`Context` object that
``NemoClient.from_config()`` uses to construct an authenticated client.

This is distinct from the server-side config in
:mod:`nemo_platform_plugin.config` which reads ``/etc/nmp/config.yaml``.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, HttpUrl, PrivateAttr, SecretStr
from typing_extensions import Self

from .models import (
    DEFAULT_BASE_URL,
    DEFAULT_CONTEXT,
    DEFAULT_WORKSPACE,
    ConfigFile,
    ConfigParams,
    Context,
    OAuthUser,
    OutputFormat,
    TimestampFormat,
)

logger = logging.getLogger(__name__)


class Config(BaseModel):
    """Configuration manager for NemoClient.

    Manages the user-facing config file (``~/.config/nmp/config.yaml``),
    runtime overrides (env vars, CLI flags, SDK params), and resolution
    of effective configuration.

    Environment variables (prefix ``NMP_``)::

        NMP_CURRENT_CONTEXT, NMP_WORKSPACE, NMP_OUTPUT_FORMAT,
        NMP_TIMESTAMP_FORMAT, NMP_COLOR_OUTPUT, NMP_BASE_URL, NMP_ACCESS_TOKEN
    """

    current_context: str | None = Field(default=None, description="Override current context (env: NMP_CURRENT_CONTEXT)")
    base_url: str | None = Field(default=None, description="Base URL for the API (env: NMP_BASE_URL)")
    access_token: SecretStr | None = Field(
        default=None,
        description="Access token for authentication (env: NMP_ACCESS_TOKEN)",
    )
    workspace: str | None = Field(default=None, description="Override workspace (env: NMP_WORKSPACE)")
    output_format: OutputFormat | None = Field(
        default=None, description="Override output format (env: NMP_OUTPUT_FORMAT)"
    )
    timestamp_format: TimestampFormat | None = Field(
        default=None, description="Override timestamp format (env: NMP_TIMESTAMP_FORMAT)"
    )
    truncate: bool | None = Field(default=None, description="Override truncate config (env: NMP_TRUNCATE)")
    color_output: bool | None = Field(default=None, description="Override color output (env: NMP_COLOR_OUTPUT)")

    _config_file: ConfigFile = PrivateAttr(default_factory=ConfigFile)
    _config_path: Path | None = PrivateAttr(default=None)

    @classmethod
    def _migrate_legacy_api_key_users(cls, config_data: dict) -> None:
        """Migrate legacy api-key users to oauth users."""
        users = config_data.get("users")
        if not isinstance(users, list):
            return

        migrated_count = 0

        for user in users:
            if not isinstance(user, dict):
                continue

            user_name = user.get("name", DEFAULT_CONTEXT)
            user_type = user.get("type")
            if user_type != "api-key" and "api_key" not in user:
                continue

            raw_value = user.get("api_key")
            if not isinstance(raw_value, str) or not raw_value.strip():
                user.clear()
                user.update({"name": user_name, "type": "no-auth"})
                migrated_count += 1
                continue

            token = raw_value.strip()
            if "@" in token:
                from nemo_platform_plugin.client.oidc import generate_unsigned_jwt

                token = generate_unsigned_jwt(
                    principal_id=token,
                    email=token,
                    expires_in_seconds=None,
                )

            user.clear()
            user.update(
                {
                    "name": user_name,
                    "type": "oauth",
                    "token": token,
                    "refresh_token": None,
                }
            )
            migrated_count += 1

        if migrated_count:
            logger.warning("Migrated %s legacy api-key user(s) to oauth users", migrated_count)

    @classmethod
    def _load_from_env(cls) -> dict[str, object]:
        """Load configuration from environment variables with NMP_ prefix."""
        env_values: dict[str, object] = {}
        for field_name in cls.model_fields:
            env_key = f"NMP_{field_name.upper()}"
            if val := os.environ.get(env_key):
                env_values[field_name] = val
        return env_values

    @classmethod
    def get_default_config_path(cls) -> Path:
        """Get the default configuration file path.

        Can be overridden with ``NMP_CONFIG_FILE`` environment variable.
        """
        env_config_path = os.environ.get("NMP_CONFIG_FILE")
        if env_config_path:
            return Path(env_config_path)

        xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config_home:
            return Path(xdg_config_home) / "nmp" / "config.yaml"

        config_dir = Path.home() / ".config" / "nmp"
        return config_dir / "config.yaml"

    @classmethod
    def create(cls, config_path: Path, config_file: ConfigFile, overrides: ConfigParams | None = None) -> Self:
        env_values = cls._load_from_env()
        merged: dict[str, object] = {**env_values}
        if overrides:
            merged.update(dict(overrides))

        config = cls.model_validate(merged)
        config._config_file = config_file
        config._config_path = config_path
        return config

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        overrides: ConfigParams | None = None,
    ) -> Self:
        """Load configuration from config file with optional parameters."""
        explicit_path_param = config_path is not None
        env_var_set = os.environ.get("NMP_CONFIG_FILE") is not None

        if config_path is None:
            config_path = cls.get_default_config_path()

        config_data = {}
        if config_path.exists():
            with open(config_path) as f:
                try:
                    config_data = yaml.safe_load(f) or {}
                except yaml.YAMLError as e:
                    raise ValueError(f"Error parsing config file at {config_path}: {e}") from e

            cls._migrate_legacy_api_key_users(config_data)
        elif explicit_path_param or env_var_set:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        config_file = ConfigFile.model_validate(config_data)
        return cls.create(config_path, config_file, overrides)

    def get_config_file(self) -> ConfigFile:
        return self._config_file

    def get_config_path(self) -> Path | None:
        return self._config_path

    def save(self, config_path: Path | None = None) -> None:
        """Save the current config file to disk."""
        path = config_path or self._config_path or self.get_default_config_path()

        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, stat.S_IRWXU)  # 700

        config_data = self._config_file.model_dump(
            mode="json",
            exclude_none=True,
            context={"include_secrets": True},
        )

        # Open with restricted permissions (0600) before writing secrets.
        # Uses os.open + os.fdopen to set permissions atomically at creation,
        # avoiding a window where the file is world-readable.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)

        self._config_path = path

    @classmethod
    def write(
        cls,
        params: ConfigParams,
        context_name: str | None = None,
        config_path: Path | None = None,
        *,
        set_current_on_create: bool = False,
    ) -> Self:
        """Write configuration settings. Creates the config file if it doesn't exist."""
        path = config_path or cls.get_default_config_path()

        if path.exists():
            config = cls.load(config_path=path)
            # Prefer explicit context_name, then params["current_context"],
            # then the currently active context from the config file.
            context_name = context_name or params.get("current_context") or config.resolve().context_name
        else:
            config = cls.create(path, ConfigFile())
            context_name = context_name or DEFAULT_CONTEXT

        config_file = config._config_file

        is_new = context_name not in {c.name for c in config_file.contexts}
        config_file.ensure_context(context_name, params)

        if "current_context" in params:
            config_file.current_context = params["current_context"]
        elif config_file.current_context is None or (set_current_on_create and is_new):
            config_file.current_context = context_name

        config.save()
        return config

    def set_current_context(self, context_name: str) -> None:
        """Set the current context in the config file and save."""
        context_names = [ctx.name for ctx in self._config_file.contexts]
        if context_name not in context_names:
            available = ", ".join(context_names) if context_names else "(none)"
            raise ValueError(f"Context '{context_name}' not found. Available contexts: {available}")

        self._config_file.current_context = context_name
        self.save()

    def reload(self) -> None:
        """Reload configuration from file (preserving runtime parameters)."""
        if self._config_path:
            current_overrides = self.get_runtime_overrides()
            new_config = self.load(config_path=self._config_path, overrides=current_overrides)

            self._config_file = new_config._config_file
            self.current_context = new_config.current_context
            self.base_url = new_config.base_url
            self.access_token = new_config.access_token
            self.workspace = new_config.workspace
            self.output_format = new_config.output_format
            self.timestamp_format = new_config.timestamp_format
            self.truncate = new_config.truncate
            self.color_output = new_config.color_output

    def resolve(self) -> Context:
        """Resolve the effective configuration by applying context and overrides."""
        if not self._config_file.contexts:
            return self._create_default_config()

        context_name = self.current_context or self._config_file.current_context or DEFAULT_CONTEXT

        context = None
        for ctx in self._config_file.contexts:
            if ctx.name == context_name:
                context = ctx
                break

        if context is None:
            available_contexts = [ctx.name for ctx in self._config_file.contexts]
            raise ValueError(f"Context '{context_name}' not found. Available contexts: {', '.join(available_contexts)}")

        cluster = None
        for clus in self._config_file.clusters:
            if clus.name == context.cluster:
                cluster = clus
                break

        if cluster is None:
            raise ValueError(f"Cluster '{context.cluster}' referenced by context '{context_name}' not found")

        user = None
        for usr in self._config_file.users:
            if usr.name == context.user:
                user = usr
                break

        if user is None:
            raise ValueError(f"User '{context.user}' referenced by context '{context_name}' not found")

        prefs = context.preferences.model_copy(deep=True)

        if self.base_url is not None:
            cluster.base_url = HttpUrl(self.base_url)
        if self.access_token is not None:
            user = OAuthUser(
                name=user.name,
                token=self.access_token,
                refresh_token=None,
            )
        if self.output_format is not None:
            prefs.output_format = self.output_format
        if self.timestamp_format is not None:
            prefs.timestamp_format = self.timestamp_format
        if self.truncate is not None:
            prefs.truncate = self.truncate
        if self.color_output is not None:
            prefs.color_output = self.color_output

        effective_workspace = self.workspace if self.workspace is not None else context.workspace
        if effective_workspace is None:
            effective_workspace = DEFAULT_WORKSPACE

        effective_default_model = os.environ.get("NEMO_DEFAULT_MODEL") or context.default_model
        effective_fast_model = os.environ.get("NEMO_FAST_MODEL") or context.fast_model or effective_default_model

        return Context(
            context_name=context_name,
            cluster=cluster,
            user=user,
            workspace=effective_workspace,
            default_model=effective_default_model,
            fast_model=effective_fast_model,
            preferences=prefs,
        )

    def get_runtime_overrides(self) -> ConfigParams:
        result: ConfigParams = {}
        result.update(self.model_dump(exclude_unset=True))
        return result

    def _create_default_config(self) -> Context:
        base_url = self.base_url or DEFAULT_BASE_URL

        params: ConfigParams = {"base_url": base_url}
        if self.access_token:
            params["access_token"] = self.access_token.get_secret_value()
        if self.workspace:
            params["workspace"] = self.workspace
        if self.output_format is not None:
            params["output_format"] = self.output_format
        if self.timestamp_format is not None:
            params["timestamp_format"] = self.timestamp_format
        if self.truncate is not None:
            params["truncate"] = self.truncate

        context_name = self.current_context or "default"

        cluster, user, context_def = self._config_file.ensure_context(context_name, params)
        self._config_file.current_context = context_name

        prefs = context_def.preferences.model_copy(deep=True)
        if self.color_output is not None:
            prefs.color_output = self.color_output

        effective_default_model = os.environ.get("NEMO_DEFAULT_MODEL") or context_def.default_model
        effective_fast_model = os.environ.get("NEMO_FAST_MODEL") or context_def.fast_model or effective_default_model

        return Context(
            context_name=context_name,
            cluster=cluster,
            user=user,
            workspace=context_def.workspace or DEFAULT_WORKSPACE,
            default_model=effective_default_model,
            fast_model=effective_fast_model,
            preferences=prefs,
        )


def get_context(
    config_path: Path | None = None,
    overrides: ConfigParams | None = None,
) -> Context:
    """Get the resolved configuration context."""
    config = Config.load(config_path=config_path, overrides=overrides)
    return config.resolve()
