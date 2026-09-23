# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration manager for nemo_helix."""

from __future__ import annotations

import logging
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

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


def _try_secure_chmod_dir(path: Path) -> None:
    """Best-effort owner-only permissions on a directory (e.g. ``/tmp`` may reject chmod)."""
    try:
        os.chmod(path, stat.S_IRWXU)  # 700
    except PermissionError:
        pass


def _write_secure_yaml(path: Path, config_data: dict) -> None:
    """Write YAML atomically with owner read/write permissions (600).

    A failure to lock down the file's permissions raises instead of being
    swallowed, so credentials are never left readable by other users.
    """
    mode = stat.S_IRUSR | stat.S_IWUSR  # 600
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "w") as f:
            fd = -1
            yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)
    finally:
        if fd >= 0:
            os.close(fd)
    # Existing files keep prior mode under O_TRUNC; enforce 600 after write.
    os.chmod(path, mode)


@dataclass(frozen=True)
class _RuntimeAccessTokenSource:
    token: str
    label: str


def _resolve_model_pair(default_model: str | None, fast_model: str | None) -> tuple[str | None, str | None]:
    """Apply environment overrides and the fast-to-default fallback consistently."""
    effective_default = os.environ.get("NEMO_DEFAULT_MODEL") or default_model
    effective_fast = os.environ.get("NEMO_FAST_MODEL") or fast_model or effective_default
    return effective_default, effective_fast


class Config(BaseModel):
    """
    Configuration manager for nemo_helix.

    This class manages:
    1. The config file (immutable source of truth)
    2. Runtime overrides (from env vars, CLI, or code)
    3. Resolution of effective configuration

    Environment variables:
    NHX_CURRENT_CONTEXT, NHX_WORKSPACE, NHX_OUTPUT_FORMAT,
    NHX_TIMESTAMP_FORMAT, NHX_PAGE_SIZE, NHX_COLOR_OUTPUT,
    NHX_BASE_URL, NHX_ACCESS_TOKEN, NEMO_DEFAULT_MODEL, NEMO_FAST_MODEL
    """

    # Runtime overrides - can be set via env vars
    current_context: str | None = Field(default=None, description="Override current context (env: NHX_CURRENT_CONTEXT)")

    base_url: str | None = Field(default=None, description="Base URL for the API (env: NHX_BASE_URL)")
    certificate_authority: str | None = Field(
        default=None,
        description="Path to a PEM certificate authority bundle for TLS verification",
    )
    access_token: SecretStr | None = Field(
        default=None,
        description="Access token for authentication (env: NHX_ACCESS_TOKEN)",
    )

    workspace: str | None = Field(default=None, description="Override workspace (env: NHX_WORKSPACE)")
    output_format: OutputFormat | None = Field(
        default=None, description="Override output format (env: NHX_OUTPUT_FORMAT)"
    )
    timestamp_format: TimestampFormat | None = Field(
        default=None, description="Override timestamp format (env: NHX_TIMESTAMP_FORMAT)"
    )
    truncate: bool | None = Field(default=None, description="Override truncate config (env: NHX_TRUNCATE)")
    color_output: bool | None = Field(default=None, description="Override color output (env: NHX_COLOR_OUTPUT)")

    # Internal state (using PrivateAttr for non-field attributes)
    _config_file: ConfigFile = PrivateAttr(default_factory=ConfigFile)
    _config_path: Path | None = PrivateAttr(default=None)

    @classmethod
    def _migrate_legacy_api_key_users(cls, config_data: dict) -> None:
        """Migrate legacy api-key users from config data to oauth users.

        Legacy users with ``type: api-key`` or ``api_key`` are converted to ``type: oauth``:
        - Email-like values are converted to an unsigned JWT (principal bootstrap compatibility)
        - Other values are treated as bearer access tokens as-is
        """
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
                # Fall back to no-auth when the legacy value is missing/invalid.
                user.clear()
                user.update(
                    {
                        "name": user_name,
                        "type": "no-auth",
                    }
                )
                migrated_count += 1
                continue

            token = raw_value.strip()
            if "@" in token:
                from nemo_helix_ext.auth.helpers import generate_unsigned_jwt

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
    def _runtime_access_token_source_from_env(cls) -> _RuntimeAccessTokenSource | None:
        if token := os.environ.get("NHX_ACCESS_TOKEN"):
            return _RuntimeAccessTokenSource(token, "NHX_ACCESS_TOKEN environment override")
        return None

    @classmethod
    def runtime_access_token_source_label(cls) -> str | None:
        """Return the effective runtime access token override source label."""
        source = cls._runtime_access_token_source_from_env()
        return source.label if source else None

    @classmethod
    def _load_from_env(cls) -> dict[str, object]:
        """Load configuration from environment variables with NHX_ prefix."""
        env_values: dict[str, object] = {}
        for field_name in cls.model_fields:
            env_key = f"NHX_{field_name.upper()}"
            if val := os.environ.get(env_key):
                env_values[field_name] = val
        if "access_token" not in env_values:
            if source := cls._runtime_access_token_source_from_env():
                env_values["access_token"] = source.token
        return env_values

    @classmethod
    def get_default_config_path(cls) -> Path:
        """
        Get the default configuration file path.

        Can be overridden with NHX_CONFIG_FILE environment variable.
        """
        # Check for env var override
        env_config_path = os.environ.get("NHX_CONFIG_FILE")
        if env_config_path:
            return Path(env_config_path)

        # Check for XDG_CONFIG_HOME environment variable
        xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config_home:
            config_dir = Path(xdg_config_home) / "nhx"
            if config_dir.exists():
                return config_dir / "config.yaml"

        # Default path
        config_dir = Path.home() / ".config" / "nhx"
        return config_dir / "config.yaml"

    @classmethod
    def create(cls, config_path: Path, config_file: ConfigFile, overrides: ConfigParams | None = None) -> Self:
        # Load env vars first (lowest priority)
        env_values = cls._load_from_env()

        # Merge: env vars < explicit overrides
        merged = {**env_values}
        if overrides:
            merged.update(overrides)

        # Create Config instance with merged values
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
        """
        Load configuration from config file with optional parameters.

        Args:
            config_path: Path to YAML config file
            overrides: Configuration parameters (these take precedence over env vars)

        Returns:
            Config instance

        Example:
            >>> config = Config.load()
            >>> config = Config.load(overrides={"current_context": "local"})
        """
        # Check if path was explicitly provided via parameter or env var
        explicit_path_param = config_path is not None
        env_var_set = os.environ.get("NHX_CONFIG_FILE") is not None

        # Determine config path
        if config_path is None:
            config_path = cls.get_default_config_path()

        # Load YAML config file
        config_data = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                try:
                    config_data = yaml.safe_load(f) or {}
                except yaml.YAMLError as e:
                    raise ValueError(f"Error parsing config file at {config_path}: {e}") from e

            cls._migrate_legacy_api_key_users(config_data)
        elif explicit_path_param or env_var_set:
            # File must exist if explicitly provided via param or env var
            raise FileNotFoundError(f"Config file not found at {config_path}")
        # Otherwise (using actual default path), missing file is okay - use empty config

        # Parse config file (immutable)
        config_file = ConfigFile.model_validate(config_data)

        return cls.create(config_path, config_file, overrides)

    def get_config_file(self) -> ConfigFile:
        """Get the immutable config file."""
        return self._config_file

    def get_config_path(self) -> Path | None:
        """Get the configuration file path."""
        return self._config_path

    def save(self, config_path: Path | None = None) -> None:
        """
        Save the current config file to disk.

        Args:
            config_path: Path to save to. If not provided, uses the path from which
                        the config was loaded, or the default path.
        """
        path = config_path or self._config_path or self.get_default_config_path()

        # Ensure parent directory exists with secure permissions (owner-only access)
        path.parent.mkdir(parents=True, exist_ok=True)
        _try_secure_chmod_dir(path.parent)

        # Serialize with secrets revealed using context
        config_data = self._config_file.model_dump(
            mode="json",
            exclude_none=True,
            context={"include_secrets": True},
        )

        _write_secure_yaml(path, config_data)

        # Update stored path if we saved to a new location
        self._config_path = path

    @classmethod
    def write(
        cls,
        params: ConfigParams,
        context_name: str | None = None,
        config_path: Path | None = None,
        *,
        set_current_on_create: bool = False,
        set_current_if_unset: bool = True,
    ) -> Self:
        """Write configuration settings and return the updated configuration."""
        return cls.write_with_result(
            params,
            context_name=context_name,
            config_path=config_path,
            set_current_on_create=set_current_on_create,
            set_current_if_unset=set_current_if_unset,
        ).config

    @classmethod
    def write_with_result(
        cls,
        params: ConfigParams,
        context_name: str | None = None,
        config_path: Path | None = None,
        *,
        set_current_on_create: bool = False,
        set_current_if_unset: bool = True,
    ) -> ConfigWriteResult[Self]:
        """
        Write configuration settings and describe the affected context.

        Creates the config file if it doesn't exist. If the context doesn't exist,
        it will be created with its own cluster and user.

        Args:
            params: Config params to apply
            context_name: Context to write; uses the current context or "default" when omitted
            config_path: Optional path override
            set_current_on_create: If True, automatically switch to the context when it is newly created.
                Superseded by set_current_if_unset for most callers; kept for backward compatibility.
            set_current_if_unset: If True, use the written context when no current context is configured

        Returns:
            Updated config, resolved context name, and whether the context was created
        """
        # 1. Determine path
        path = config_path or cls.get_default_config_path()
        context_name_was_explicit = context_name is not None

        # 2. Load existing or create empty
        if path.exists():
            config = cls.load(config_path=path)
            if not context_name_was_explicit and config.current_context is not None:
                available_contexts = [context.name for context in config.get_config_file().contexts]
                if config.current_context not in available_contexts:
                    raise ValueError(
                        f"Context '{config.current_context}' not found. "
                        f"Available contexts: {', '.join(available_contexts)}"
                    )
            # Respect NHX_CURRENT_CONTEXT (config.current_context) before the stored value.
            # Avoid config.resolve() here: it raises when contexts exist but current_context is None.
            context_name = (
                context_name or config.current_context or config.get_config_file().current_context or DEFAULT_CONTEXT
            )
        else:
            config = cls.create(path, ConfigFile())
            context_name = context_name or config.current_context or DEFAULT_CONTEXT

        config_file = config._config_file

        # 3. Check if the context already exists (before ensure_context may create it)
        is_new = context_name not in {c.name for c in config_file.contexts}

        # 4. Find or create entities to update
        config_file.ensure_context(context_name, params)

        # 5. Set current_context
        if "current_context" in params:
            config_file.current_context = params["current_context"]
        elif (set_current_if_unset and config_file.current_context is None) or (set_current_on_create and is_new):
            config_file.current_context = context_name

        # 6. Save
        config.save()

        return ConfigWriteResult(config=config, context_name=context_name, created=is_new)

    def set_current_context(self, context_name: str) -> None:
        """
        Set the current context in the config file and save.

        Args:
            context_name: Name of the context to set as current.

        Raises:
            ValueError: If the context does not exist.
        """
        # Validate context exists
        context_names = [ctx.name for ctx in self._config_file.contexts]
        if context_name not in context_names:
            available = ", ".join(context_names) if context_names else "(none)"
            raise ValueError(f"Context '{context_name}' not found. Available contexts: {available}")

        # Update the config file model
        self._config_file.current_context = context_name

        # Save to disk
        self.save()

    @classmethod
    def delete_context(
        cls,
        context_name: str,
        config_path: Path | None = None,
        *,
        prune_orphans: bool = False,
    ) -> ConfigDeleteContextResult[Self]:
        """Delete a context definition from the config file.

        This follows kubectl's convention: deleting a context only removes the
        context reference. Referenced cluster and user records are left intact
        unless ``prune_orphans`` is explicitly requested.
        """
        path = config_path or cls.get_default_config_path()
        config = cls.load(config_path=path)
        config_file = config._config_file

        context_index = next(
            (index for index, context in enumerate(config_file.contexts) if context.name == context_name),
            None,
        )
        if context_index is None:
            available = (
                ", ".join(context.name for context in config_file.contexts) if config_file.contexts else "(none)"
            )
            raise ValueError(f"Context '{context_name}' not found. Available contexts: {available}")

        del config_file.contexts[context_index]

        current_context_cleared = config_file.current_context == context_name
        if current_context_cleared:
            config_file.current_context = None

        pruned_clusters: list[str] = []
        pruned_users: list[str] = []
        if prune_orphans:
            referenced_clusters = {context.cluster for context in config_file.contexts}
            referenced_users = {context.user for context in config_file.contexts}

            pruned_clusters = [
                cluster.name for cluster in config_file.clusters if cluster.name not in referenced_clusters
            ]
            pruned_users = [user.name for user in config_file.users if user.name not in referenced_users]

            config_file.clusters = [cluster for cluster in config_file.clusters if cluster.name in referenced_clusters]
            config_file.users = [user for user in config_file.users if user.name in referenced_users]

        config.save()
        return ConfigDeleteContextResult(
            config=config,
            context_name=context_name,
            current_context_cleared=current_context_cleared,
            pruned_clusters=pruned_clusters,
            pruned_users=pruned_users,
        )

    def reload(self) -> None:
        """Reload configuration from file (preserving runtime parameters)."""
        if self._config_path:
            # Save current parameters
            current_overrides = self.get_runtime_overrides()

            # Reload
            new_config = self.load(config_path=self._config_path, overrides=current_overrides)

            # Update state
            self._config_file = new_config._config_file
            self.current_context = new_config.current_context

            self.base_url = new_config.base_url
            self.certificate_authority = new_config.certificate_authority
            self.access_token = new_config.access_token

            self.workspace = new_config.workspace

            self.output_format = new_config.output_format
            self.timestamp_format = new_config.timestamp_format
            self.truncate = new_config.truncate
            self.color_output = new_config.color_output

    def resolve(self) -> Context:
        """
        Resolve the effective configuration by applying context and overrides.

        This method transforms a ContextDefinition (from the config file) into a
        Context (runtime model) by:
        1. Resolving the cluster and user name references into full objects
        2. Applying runtime overrides (env vars, CLI flags, SDK params)
        3. Merging preferences from multiple sources

        Returns:
            Context with effective configuration

        Raises:
            ValueError: If context, cluster, or user not found, or if required configuration is missing
        """
        # If no contexts are defined, use direct configuration from env vars
        if not self._config_file.contexts:
            return self._create_default_config()

        # Normal path: contexts are defined in config file
        # Determine effective context name
        context_name = self.current_context or self._config_file.current_context

        # Find the ContextDefinition from the config file
        context = None
        for ctx in self._config_file.contexts:
            if ctx.name == context_name:
                context = ctx
                break

        if context is None:
            available_contexts = [ctx.name for ctx in self._config_file.contexts]
            raise ValueError(f"Context '{context_name}' not found. Available contexts: {', '.join(available_contexts)}")

        # Resolve the cluster reference (string) into a full Cluster object
        cluster = None
        for clus in self._config_file.clusters:
            if clus.name == context.cluster:
                cluster = clus
                break

        if cluster is None:
            raise ValueError(f"Cluster '{context.cluster}' referenced by context '{context_name}' not found")
        cluster = cluster.model_copy(deep=True)

        # Resolve the user reference (string) into AuthConfig
        user = None
        for usr in self._config_file.users:
            if usr.name == context.user:
                user = usr
                break

        if user is None:
            raise ValueError(f"User '{context.user}' referenced by context '{context_name}' not found")

        # Build effective preferences
        prefs = context.preferences.model_copy(deep=True)

        # Apply runtime overrides
        if self.base_url is not None:
            cluster.base_url = HttpUrl(self.base_url)
        if self.certificate_authority is not None:
            cluster.certificate_authority = self.certificate_authority
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

        # Determine effective workspace (always provide a default)
        effective_workspace = self.workspace if self.workspace is not None else context.workspace
        if effective_workspace is None:
            effective_workspace = DEFAULT_WORKSPACE

        # The default model is the quality-oriented model for agent workloads.
        # Existing single-model contexts also supply the fast role.
        effective_default_model, effective_fast_model = _resolve_model_pair(
            context.default_model,
            context.fast_model,
        )

        # Create the resolved Context (runtime model) from the ContextDefinition (config file model)
        return Context(
            context_name=context_name,
            cluster=cluster,  # Fully resolved Cluster object, not just a string reference
            user=user,  # Fully resolved User with authentication credentials
            workspace=effective_workspace,
            default_model=effective_default_model,
            fast_model=effective_fast_model,
            preferences=prefs,
        )

    def get_runtime_overrides(self) -> ConfigParams:
        """Get current runtime parameters."""
        result: ConfigParams = {}
        # update the result dictionary with the current runtime parameters
        result.update(self.model_dump(exclude_unset=True))
        return result

    def _create_default_config(self) -> Context:
        """
        Create a default configuration when no config file is present.

        Uses direct configuration from environment variables or SDK parameters.

        Returns:
            Context with default settings

        Raises:
            ValueError: If required configuration (base_url) is missing
        """
        base_url = self.base_url or DEFAULT_BASE_URL

        # Build params from runtime overrides
        params: ConfigParams = {"base_url": base_url}
        if self.certificate_authority is not None:
            params["certificate_authority"] = self.certificate_authority
        if self.access_token:
            params["access_token"] = self.access_token.get_secret_value()
        if self.workspace:
            params["workspace"] = self.workspace
        if self.output_format:
            params["output_format"] = self.output_format
        if self.timestamp_format:
            params["timestamp_format"] = self.timestamp_format
        if self.truncate:
            params["truncate"] = self.truncate

        context_name = self.current_context or "default"

        # Use ConfigFile method to create entities
        cluster, user, context_def = self._config_file.ensure_context(context_name, params)
        self._config_file.current_context = context_name

        # Apply color_output override (not in ConfigParams)
        prefs = context_def.preferences.model_copy(deep=True)
        if self.color_output is not None:
            prefs.color_output = self.color_output

        # Return resolved Context (runtime model)
        effective_default_model, effective_fast_model = _resolve_model_pair(
            context_def.default_model,
            context_def.fast_model,
        )
        return Context(
            context_name=context_name,
            cluster=cluster,
            user=user,
            workspace=context_def.workspace or DEFAULT_WORKSPACE,
            default_model=effective_default_model,
            fast_model=effective_fast_model,
            preferences=prefs,
        )


_T = TypeVar("_T", bound=Config)


@dataclass(frozen=True)
class ConfigWriteResult(Generic[_T]):
    """Result metadata for a configuration write."""

    config: _T
    context_name: str
    created: bool


@dataclass(frozen=True)
class ConfigDeleteContextResult(Generic[_T]):
    """Result metadata for a context deletion."""

    config: _T
    context_name: str
    current_context_cleared: bool
    pruned_clusters: list[str]
    pruned_users: list[str]


def get_context(
    config_path: Path | None = None,
    overrides: ConfigParams | None = None,
) -> Context:
    """
    Get the resolved configuration context.

    This is a simple stateless helper that loads configuration and returns
    a resolved Context object. Use dependency injection to pass the Context
    to application code.

    Args:
        config_path: Path to config file (defaults to ~/.config/nhx/config.yaml)
        overrides: Configuration parameters to override

    Returns:
        Context with effective configuration

    Example:
        >>> context = get_context()
        >>> print(f"Using context: {context.context_name}")
        >>> print(f"Cluster URL: {context.cluster.base_url}")
        >>> print(f"Workspace: {context.workspace}")
    """
    config = Config.load(config_path=config_path, overrides=overrides)
    return config.resolve()
