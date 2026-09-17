# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage configuration classes for various backends.

These configs can be used by any service that needs to interact with storage backends.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import (
    Annotated,
    Literal,
    Self,
)

from nemo_platform_plugin.config import nmp_user_data_dir
from nemo_platform_plugin.schema import SecretRef
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class StorageConfigType(StrEnum):
    LOCAL = "local"
    NGC = "ngc"
    HUGGINGFACE = "huggingface"
    S3 = "s3"
    GITHUB = "github"
    # AZURE_BLOB = "azure_blob"
    # GCS = "gcs"
    # HTTP = "http"


# Default chunk size for reading/streaming files (1MB)
DEFAULT_READ_CHUNK_SIZE = 1 * 1024 * 1024


def _tracked_revision(revision: str, original_revision: str | None) -> str | None:
    """Return the ref *revision* was resolved from, when that ref can still move.

    Resolution records the original revision even when the user pinned an
    immutable id themselves, and a ref equal to what it resolved to cannot name
    anything else.
    """
    return original_revision if original_revision and original_revision != revision else None


def _reject_blank(field: str, value: str) -> str:
    """Refuse a value that would drop out of a URL built from it.

    An empty segment is skipped when the path is joined, so a blank revision turns
    ``/commits/{revision}`` into the list-commits endpoint rather than failing.
    """
    if not value.strip():
        raise ValueError(f"{field} must not be blank")
    return value


def _reject_relative_segments(field: str, value: str) -> str:
    """Refuse values that would re-point a URL built from them at another resource.

    ``..`` is resolved away by the URL layer before the request is sent, so a
    dot segment escapes the repository the rest of the config names.
    """
    if any(segment in (".", "..") for segment in value.split("/")):
        raise ValueError(f"{field} must not contain '.' or '..' path segments, got {value!r}")
    return value


class BaseStorageConfig(BaseModel):
    read_chunk_size: int = Field(
        default=DEFAULT_READ_CHUNK_SIZE,
        description="Chunk size in bytes for reading/streaming files. "
        "Larger chunks reduce async overhead but increase memory per concurrent download. "
        "Default: 1MB.",
    )

    def get_secret_references(self) -> dict[str, SecretRef]:
        """Get the secret references for the storage config."""
        return {}

    @property
    def pinned_revision(self) -> str:
        """The immutable id this storage is pinned to, empty when it pins nothing."""
        return ""

    @property
    def tracked_revision(self) -> str | None:
        """The mutable ref :attr:`pinned_revision` was resolved from, if it can still move.

        None when the fileset was created from an already-immutable id, which has
        nothing to move to.
        """
        return None

    @property
    def owns_storage_data(self) -> bool:
        """Whether the platform owns the underlying source data for this backend.

        When True, deleting a fileset must also delete the underlying source
        data (e.g. local files, S3 objects under our prefix). When False, the
        backend points at source data the platform does not own and must not
        delete (e.g. read-only external registries like NGC or HuggingFace).

        Defaults to False so external backends are safe by default.
        """
        return False

    def copy_config(self, path: str) -> Self:
        """
        This method is necessary for when we're using a storage config
        as the default storage config. We will create a new fileset that takes
        the config-defined storage config and create a fileset within a subpath of
        that storage config.

        Only specific backends will be able to support this functionality,
        so by default we should raise an error.
        """
        raise NotImplementedError()


class LocalStorageConfig(BaseStorageConfig):
    type: Literal[StorageConfigType.LOCAL] = StorageConfigType.LOCAL
    path: str

    # These flags below will likely never be used by end-users, but they're useful
    # during iteration to fine-tune performance.
    write_buffer_size: int = Field(
        default=16 * 1024 * 1024,
        description="How many bytes to buffer before flushing to disk",
    )

    @field_validator("path")
    @classmethod
    def make_path_relative_to_program(cls, v: str) -> str:
        """
        This allows the config to pass in absolute paths, ``~``-prefixed
        paths (expanded against the running user's home dir), or relative
        paths like ``./files_storage`` (joined against cwd).

        An **empty** path means the platform's user-data directory, so blobs follow
        ``NMP_DATA_DIR`` alongside the entity-store database.

        Resolved here rather than as a field default because a machine-dependent default is
        rendered into the committed config reference — the same reason the SQLite path is
        computed in ``get_database_url``.
        """
        if not v:
            return str(nmp_user_data_dir() / "files")
        return str(Path.cwd() / Path(v).expanduser())

    @property
    def owns_storage_data(self) -> bool:
        # Deleting a local-backed fileset removes the underlying directory
        # (see LocalStorageImpl.delete_all), so we own that data.
        return True

    def copy_config(self, path: str) -> Self:
        new_subpath = os.path.join(self.path, path)
        return self.model_copy(deep=True, update={"path": new_subpath})


class HuggingfaceStorageConfig(BaseStorageConfig):
    type: Literal[StorageConfigType.HUGGINGFACE] = StorageConfigType.HUGGINGFACE
    repo_id: str = Field(description="Huggingface repository ID (e.g., 'meta-llama/Llama-2-7b')")
    repo_type: Literal["model", "dataset", "space"] = Field(
        default="model",
        description="Type of Huggingface repository: 'model', 'dataset', or 'space'",
    )
    revision: str = Field(
        default="main",
        description="Branch, tag, or commit SHA. Defaults to 'main'",
    )
    original_revision: str | None = Field(
        default=None,
        description="The original revision requested by the user before resolution (e.g., 'main'). "
        "The 'revision' field contains the resolved commit SHA.",
    )

    token_secret: SecretRef | None = Field(
        default=None,
        description="Huggingface API `token` secret name for private repositories",
    )

    endpoint: str = Field(
        default="https://huggingface.co",
        description="Huggingface Hub endpoint URL. Use for self-hosted instances.",
    )

    @property
    def pinned_revision(self) -> str:
        return self.revision

    @property
    def tracked_revision(self) -> str | None:
        return _tracked_revision(self.revision, self.original_revision)

    def get_secret_references(self) -> dict[str, SecretRef]:
        return {"token": self.token_secret} if self.token_secret else {}


class GithubStorageConfig(BaseStorageConfig):
    type: Literal[StorageConfigType.GITHUB] = StorageConfigType.GITHUB
    owner: str = Field(description="GitHub repository owner (user or organization)")
    repo: str = Field(description="GitHub repository name")
    revision: str = Field(
        default="HEAD",
        description="Branch, tag, or commit SHA. 'HEAD' resolves to the repository's default branch.",
    )
    original_revision: str | None = Field(
        default=None,
        description="The original revision requested by the user before resolution (e.g., 'main'). "
        "The 'revision' field contains the resolved commit SHA.",
    )
    path: str = Field(
        default="",
        description="Optional directory within the repository. All paths are relative to it.",
    )

    token_secret: SecretRef | None = Field(
        default=None,
        description="GitHub personal access token secret name, required for private repositories",
    )

    api_base_url: str = Field(
        default="https://api.github.com",
        description="GitHub API base URL. Use for GitHub Enterprise instances.",
    )

    @field_validator("path")
    @classmethod
    def strip_path_slashes(cls, v: str) -> str:
        return _reject_relative_segments("path", v.strip("/"))

    @field_validator("owner", "repo")
    @classmethod
    def reject_multi_segment_names(cls, v: str, info: ValidationInfo) -> str:
        field = info.field_name or "value"
        if "/" in v:
            raise ValueError(f"{field} must name a single path segment, got {v!r}")
        return _reject_relative_segments(field, _reject_blank(field, v))

    @field_validator("revision")
    @classmethod
    def reject_relative_revision(cls, v: str) -> str:
        return _reject_relative_segments("revision", _reject_blank("revision", v))

    @field_validator("api_base_url")
    @classmethod
    def require_https(cls, v: str) -> str:
        # Every request to this host carries the token, and the external-host
        # allowlist matches on scheme, so an allowlisted http:// host would send it
        # in cleartext.
        if not v.lower().startswith("https://"):
            raise ValueError(f"api_base_url must use https, got {v!r}")
        return v

    @property
    def pinned_revision(self) -> str:
        return self.revision

    @property
    def tracked_revision(self) -> str | None:
        return _tracked_revision(self.revision, self.original_revision)

    def get_secret_references(self) -> dict[str, SecretRef]:
        return {"token": self.token_secret} if self.token_secret else {}


class NGCStorageConfig(BaseStorageConfig):
    type: Literal[StorageConfigType.NGC] = StorageConfigType.NGC
    org: str = Field(description="NGC organization name")
    team: str = Field(description="NGC team name")
    target: str = Field(description="NGC asset name (model or resource)")
    target_type: Literal["resource", "model"] = Field(
        default="resource",
        description="Type of NGC asset: 'resource' or 'model'",
    )
    version: str | None = Field(
        default=None,
        description="NGC asset version. If not provided, defaults to latest version",
    )
    original_version: str | None = Field(
        default=None,
        description="The original version requested by the user before resolution (e.g., 'latest' or None). "
        "The 'version' field contains the resolved version ID.",
    )

    api_key_secret: SecretRef = Field(description="NGC API key secret name")

    host: str = Field(
        default="https://api.ngc.nvidia.com",
        description="NGC API host URL",
    )

    def get_secret_references(self) -> dict[str, SecretRef]:
        return {"api_key": self.api_key_secret}


class S3StorageConfig(BaseStorageConfig):
    type: Literal[StorageConfigType.S3] = StorageConfigType.S3
    bucket: str = Field(description="S3 bucket name")
    prefix: str = Field(
        default="",
        description="Optional prefix (folder path) within the bucket. All operations will be relative to this prefix.",
    )
    region: str | None = Field(
        default=None,
        description="AWS region. If not specified, uses SDK default (env vars, instance metadata, etc.)",
    )
    endpoint_url: str | None = Field(
        default=None,
        description="Custom endpoint URL for S3-compatible storage (e.g., MinIO, Garage, RustFS). "
        "If not specified, uses AWS S3.",
    )
    use_sdk_auth: bool = Field(
        default=False,
        description="Use AWS SDK credential chain for authentication (env vars like AWS_ACCESS_KEY_ID, "
        "IAM roles, instance profiles, etc.). This option is only available for the platform's default "
        "storage backend. User-provided S3 storage must use explicit credentials via "
        "access_key_id_secret and secret_access_key_secret.",
    )
    access_key_id_secret: SecretRef | None = Field(
        default=None,
        description="Secret reference for AWS access key ID. Requires use_sdk_auth=False.",
    )
    secret_access_key_secret: SecretRef | None = Field(
        default=None,
        description="Secret reference for AWS secret access key. Requires use_sdk_auth=False.",
    )
    signature_version: Literal["s3v4", "s3"] = Field(
        default="s3v4",
        description="AWS signature version for request signing. "
        "Use 's3' for legacy systems that only support signature v2.",
    )

    @model_validator(mode="after")
    def validate_auth_config(self) -> Self:
        """Validate auth configuration is consistent."""
        has_secrets = self.access_key_id_secret is not None or self.secret_access_key_secret is not None

        if self.use_sdk_auth and has_secrets:
            raise ValueError(
                "use_sdk_auth=True is mutually exclusive with access_key_id_secret and "
                "secret_access_key_secret. Set use_sdk_auth=False to use explicit credentials."
            )

        if not self.use_sdk_auth:
            if self.access_key_id_secret is None or self.secret_access_key_secret is None:
                raise ValueError(
                    "Both access_key_id_secret and secret_access_key_secret must be provided when use_sdk_auth=False."
                )

        return self

    def get_secret_references(self) -> dict[str, SecretRef]:
        refs: dict[str, SecretRef] = {}
        if self.access_key_id_secret:
            refs["access_key_id"] = self.access_key_id_secret
        if self.secret_access_key_secret:
            refs["secret_access_key"] = self.secret_access_key_secret
        return refs

    @property
    def owns_storage_data(self) -> bool:
        # Deleting an S3-backed fileset removes the objects under our prefix
        # (see S3StorageImpl.delete_all), so we own that source data.
        return True

    def copy_config(self, path: str) -> Self:
        """Create a copy with an extended prefix for subpath filesets."""
        new_prefix = f"{self.prefix.rstrip('/')}/{path}" if self.prefix else path
        return self.model_copy(deep=True, update={"prefix": new_prefix})


StorageConfig = LocalStorageConfig | NGCStorageConfig | HuggingfaceStorageConfig | S3StorageConfig | GithubStorageConfig

StorageConfigField = Annotated[StorageConfig, Field(discriminator="type")]
