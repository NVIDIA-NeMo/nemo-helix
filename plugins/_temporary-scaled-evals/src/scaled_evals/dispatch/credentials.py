# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch-time BYOK credential materialization.

Evaluation rows store role -> credential id references. The dispatcher resolves
those references once, decrypts the payloads, and passes only an ephemeral env
map to runtime backends. API read paths continue to return metadata only.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from scaled_evals.api import crypto
from scaled_evals.api.repositories.credential_repository import CredentialRepository
from scaled_evals.api.settings import settings

_PROVIDER_ENV_KEYS = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    # Recipe 12 maps the same inference key to OPENAI_API_KEY for OpenAI-wire agents.
    "nvidia": ("NGC_INFERENCE_API_KEY", "POLICY_API_KEY", "OPENAI_API_KEY"),
}


def _env_file_value(value: str) -> str:
    if not value:
        raise ValueError("credential environment values must be non-empty strings")
    if any(ch.isspace() for ch in value) or any(ch in value for ch in "#='\""):
        escaped = value.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def write_env_file(path: Path, env: Mapping[str, str]) -> None:
    """Write an env file from already-materialized values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write("".join(f"{key}={_env_file_value(value)}\n" for key, value in env.items()))


def credential_env_from_rows(rows: list[Mapping[str, object]]) -> dict[str, str]:
    """Convert credential rows into runtime env vars."""
    plaintext = {
        str(row["id"]): crypto.decrypt(row["encrypted_payload"])  # type: ignore[arg-type]
        for row in rows
    }
    return _credential_env_from_plaintext_rows(rows, plaintext)


def _credential_env_from_plaintext_rows(
    rows: list[Mapping[str, object]],
    plaintext_by_id: Mapping[str, str],
) -> dict[str, str]:
    env: dict[str, str] = {}
    for row in rows:
        provider = str(row["provider"])
        payload_kind = str(row["payload_kind"])
        plaintext = plaintext_by_id[str(row["id"])]
        if provider in _PROVIDER_ENV_KEYS:
            if payload_kind != "key":
                raise ValueError(f"credential {row['id']} for {provider} must use key payload")
            for key in _PROVIDER_ENV_KEYS[provider]:
                env[key] = plaintext
            if provider == "anthropic":
                env.setdefault("ANTHROPIC_BASE_URL", settings.nvidia_anthropic_base_url)
            elif provider == "nvidia":
                env.setdefault("POLICY_BASE_URL", settings.nvidia_inference_base_url)
        elif provider == "nhx":
            env["NHX_INTAKE_CREDENTIAL_YAML"] = plaintext
        elif provider == "openshift":
            # A user's OpenShift bearer token. NOT a runtime env var for the
            # agent — sandbox_k8s extracts it to write a per-eval kubeconfig and
            # strips it before the agent env-file, so it never reaches the pod.
            if payload_kind != "key":
                raise ValueError(f"credential {row['id']} for openshift must use key payload")
            env["SANDBOX_OC_TOKEN"] = plaintext
        else:
            raise ValueError(f"unsupported credential provider: {provider}")
    return env


def materialize_credential_env(  # noqa: ANN001
    conn,
    credentials: Mapping[str, str],
    *,
    owner_id: str | None = None,
    include_unowned: bool = False,
    expected: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, str]:
    """Resolve role -> credential-id refs into decrypted runtime env vars."""
    cred_ids = sorted(set(credentials.values()))
    if not cred_ids:
        return {}
    rows = CredentialRepository(conn).load_for_dispatch(
        cred_ids,
        owner_id=owner_id,
        include_unowned=include_unowned,
    )
    rows_by_id = {row["id"]: row for row in rows}
    missing = [cred_id for cred_id in cred_ids if cred_id not in rows_by_id]
    if missing:
        raise ValueError(f"credential not found: {missing[0]}")
    for credential_id, expectation in (expected or {}).items():
        row = rows_by_id.get(credential_id)
        if row is None:
            raise ValueError(f"snapshotted credential not found: {credential_id}")
        for field in ("fingerprint", "provider", "payload_kind"):
            actual = str(row.get(field) or "")
            wanted = str(expectation.get(field) or "")
            if not wanted or actual != wanted:
                raise ValueError(f"credential {credential_id} {field} changed after evaluation submission")
    providers = [str(rows_by_id[cred_id]["provider"]) for cred_id in cred_ids]
    duplicate = next((provider for provider in providers if providers.count(provider) > 1), None)
    if duplicate:
        # Each provider fills fixed env keys, so a second credential would silently replace the first.
        raise ValueError(f"multiple {duplicate} credentials are not supported")
    return credential_env_from_rows([rows_by_id[cred_id] for cred_id in cred_ids])


def merged_env_file(
    *,
    source_env_file: Path,
    output_env_file: Path,
    credential_env: Mapping[str, str],
) -> Path:
    """Create an evaluation-scoped env file with credential overrides."""
    if not credential_env:
        return source_env_file
    merged = source_env_file.read_text()
    if merged and not merged.endswith("\n"):
        merged += "\n"
    output_env_file.parent.mkdir(parents=True, exist_ok=True)
    output_env_file.write_text(merged)
    with output_env_file.open("a") as handle:
        for key, value in credential_env.items():
            handle.write(f"{key}={_env_file_value(value)}\n")
    output_env_file.chmod(0o600)
    return output_env_file
