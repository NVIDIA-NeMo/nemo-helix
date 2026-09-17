# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persona-management subcommands for ``nemo data-designer personas``."""

from __future__ import annotations

import os
from typing import Annotated

import click
import typer
from data_designer.cli.ui import print_error, print_header, print_success
from data_designer_nemo.nemotron_personas import (
    SUPPORTED_LOCALES,
    WORKSPACE,
    get_resource_name_for_locale,
    sync_nemotron_personas_fileset,
)
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.client.errors import ConflictError
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.secrets.client import SecretsClient
from nemo_platform_plugin.secrets.types import PlatformSecretCreateRequest
from pydantic import SecretStr

_SUPPORTED_LOCALE_NAMES = sorted(SUPPORTED_LOCALES)


def _parse_api_key_secret(api_key_secret: str) -> tuple[str, str]:
    parts = api_key_secret.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise typer.BadParameter(
            "must be fully qualified as WORKSPACE/NAME",
            param_hint="--api-key-secret",
        )
    return parts[0], parts[1]


def _get_api_key_from_env(api_key_env_var: str | None) -> str | None:
    if api_key_env_var is None:
        return None

    api_key = os.environ.get(api_key_env_var)
    if not api_key:
        raise typer.BadParameter(
            f"environment variable {api_key_env_var!r} is not set or is empty",
            param_hint="--api-key-env-var",
        )
    return api_key


def make_fileset_command(
    locale: Annotated[
        str,
        typer.Option(
            "--locale",
            help="Locale fileset to create.",
            click_type=click.Choice(_SUPPORTED_LOCALE_NAMES),
        ),
    ],
    api_key_secret: Annotated[
        str,
        typer.Option(
            "--api-key-secret",
            help="Fully qualified NGC API key secret reference (WORKSPACE/NAME).",
        ),
    ],
    api_key_env_var: Annotated[
        str | None,
        typer.Option(
            "--api-key-env-var",
            help="Environment variable containing an NGC API key to create at --api-key-secret.",
        ),
    ] = None,
) -> None:
    """Create the system fileset for one Nemotron Personas locale."""
    if locale not in SUPPORTED_LOCALES:
        raise typer.BadParameter(
            f"unsupported locale {locale!r}; choose from {', '.join(_SUPPORTED_LOCALE_NAMES)}",
            param_hint="--locale",
        )

    secret_workspace, secret_name = _parse_api_key_secret(api_key_secret)
    api_key = _get_api_key_from_env(api_key_env_var)

    print_header("Nemotron Personas Fileset")
    client = NemoClient.from_config()
    files = FilesClient.from_client(client)

    if api_key is not None:
        try:
            secrets = SecretsClient.from_client(client)
            secrets.create_secret(
                workspace=secret_workspace,
                body=PlatformSecretCreateRequest(name=secret_name, value=SecretStr(api_key)),
            )
        except ConflictError as exc:
            print_error(
                f"Secret {api_key_secret!r} already exists. Omit --api-key-env-var to reuse an existing secret."
            )
            raise typer.Exit(code=1) from exc
        except Exception as exc:
            print_error(f"Failed to create secret {api_key_secret!r}: {exc}")
            raise typer.Exit(code=1) from exc
        print_success(f"Created secret {api_key_secret!r}")

    fileset_name = get_resource_name_for_locale(locale)
    fileset_ref = f"{WORKSPACE}/{fileset_name}"
    try:
        result = sync_nemotron_personas_fileset(files=files, locale=locale, api_key_secret=api_key_secret)
    except Exception as exc:
        print_error(f"Failed to create fileset {fileset_ref!r}: {exc}")
        raise typer.Exit(code=1) from exc

    if result == "created":
        print_success(f"Created fileset {fileset_ref!r} for locale {locale!r}")
    else:
        print_success(f"Fileset {fileset_ref!r} already exists for locale {locale!r}")
