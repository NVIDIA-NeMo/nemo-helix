# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for locating and building the embedded OPA WASM policy asset."""

import logging
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

AUTH_PACKAGE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_WASM_PATH = AUTH_PACKAGE_DIR / "assets" / "policy.wasm"
DEFAULT_BUILD_TIMEOUT_SECONDS = 120
OPA_VERSION = os.environ.get("OPA_VERSION", "v1.8.0")
OPA_VERSION_NO_V = OPA_VERSION.removeprefix("v")


class PolicyWasmError(RuntimeError):
    """Raised when the embedded PDP WASM artifact cannot be prepared."""


def discover_repo_root(start: Path | None = None) -> Path | None:
    """Return the NeMo Helix source checkout root when running from source."""
    current = (start or Path(__file__).resolve()).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        build_script = candidate / "script" / "build_policy_wasm.sh"
        policy_dir = candidate / "services" / "core" / "auth" / "src" / "nhx" / "core" / "auth" / "app" / "policies"
        if build_script.is_file() and policy_dir.is_dir():
            return candidate
    return None


def policy_wasm_needs_build(
    *,
    wasm_path: Path = DEFAULT_POLICY_WASM_PATH,
) -> bool:
    """Return True when policy.wasm is missing."""
    return not wasm_path.exists()


def _opa_asset_name() -> str:
    if os.uname().sysname.lower() == "darwin":
        os_name = "darwin"
    else:
        os_name = "linux"

    machine = os.uname().machine.lower()
    if machine in {"x86_64", "amd64"}:
        arch = "amd64"
    elif machine in {"aarch64", "arm64"}:
        arch = "arm64"
    else:
        arch = machine
    return f"opa_{os_name}_{arch}_static"


def _offline_build_hint(repo_root: Path, wasm_path: Path) -> str:
    asset = _opa_asset_name()
    cache_path = repo_root / ".cache" / "opa" / OPA_VERSION / asset
    return (
        "\n\nOffline remediation options:\n"
        f"  1. Provide a local OPA {OPA_VERSION} binary and rerun startup:\n"
        f"       OPA_BIN=/path/to/{asset} uv run nemo services run --host 127.0.0.1 --port 8080\n"
        "\n"
        "  2. Seed the script cache and rerun startup:\n"
        f"       mkdir -p {cache_path.parent}\n"
        f"       cp /path/to/{asset} {cache_path}\n"
        f"       chmod +x {cache_path}\n"
        "\n"
        "  3. If you only need to test startup with an already-built artifact, copy a current policy.wasm:\n"
        f"       cp /path/to/policy.wasm {wasm_path}\n"
        "\n"
        f'The OPA binary must report "Version: {OPA_VERSION_NO_V}" from `/path/to/{asset} version`.'
    )


def _build_policy_wasm(
    *,
    repo_root: Path,
    wasm_path: Path,
    timeout_seconds: int,
    env_overrides: Mapping[str, str] | None,
) -> None:
    build_script = repo_root / "script" / "build_policy_wasm.sh"
    if not build_script.is_file():
        raise PolicyWasmError(
            f"Embedded auth PDP policy.wasm is missing at {wasm_path}, "
            f"but the build script was not found at {build_script}." + _offline_build_hint(repo_root, wasm_path)
        )

    build_env = os.environ.copy()
    build_env.update(
        {
            "REPO_ROOT": str(repo_root),
            "OUTPUT_DIR": str(wasm_path.parent),
        }
    )
    if env_overrides:
        build_env.update(env_overrides)

    logger.info(
        "Building embedded auth PDP policy.wasm", extra={"repo_root": str(repo_root), "wasm_path": str(wasm_path)}
    )
    result = subprocess.run(
        [str(build_script)],
        cwd=repo_root,
        env=build_env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part) or "(no output)"
        hint = "" if "Unable to prepare OPA" in output else _offline_build_hint(repo_root, wasm_path)
        raise PolicyWasmError(
            "Failed to build embedded auth PDP policy.wasm.\n\n"
            "Command:\n"
            "  script/build_policy_wasm.sh\n\n"
            f"Exit code: {result.returncode}\n\n"
            "Build output:\n"
            f"{output}" + hint
        )

    if not wasm_path.exists():
        raise PolicyWasmError(
            f"script/build_policy_wasm.sh completed successfully but did not create policy.wasm at {wasm_path}."
        )


def ensure_embedded_policy_wasm(
    *,
    auto_build: bool = True,
    wasm_path: Path = DEFAULT_POLICY_WASM_PATH,
    repo_root: Path | None = None,
    discover_source_checkout: bool = True,
    build_timeout_seconds: int = DEFAULT_BUILD_TIMEOUT_SECONDS,
    env_overrides: Mapping[str, str] | None = None,
) -> Path:
    """Ensure the embedded PDP WASM artifact exists."""
    resolved_repo_root = repo_root
    if resolved_repo_root is None and discover_source_checkout:
        resolved_repo_root = discover_repo_root()

    if not policy_wasm_needs_build(wasm_path=wasm_path):
        return wasm_path

    if not auto_build:
        raise PolicyWasmError(
            f"Embedded auth PDP policy.wasm is missing at {wasm_path}. "
            "Run `make build-policy` from the NeMo Helix repo root, or set "
            "`auth.embedded_pdp_auto_build_wasm=true` for local source checkouts."
        )

    if resolved_repo_root is None:
        raise PolicyWasmError(
            f"Embedded auth PDP policy.wasm is missing at {wasm_path}, "
            "and this does not look like a NeMo Helix source checkout. Rebuild the package/image "
            "with the policy WASM artifact included."
        )

    _build_policy_wasm(
        repo_root=resolved_repo_root,
        wasm_path=wasm_path,
        timeout_seconds=build_timeout_seconds,
        env_overrides=env_overrides,
    )
    return wasm_path
