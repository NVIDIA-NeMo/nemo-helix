# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive prompts for quickstart configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import SecretStr
from rich.console import Console

from nemo_helix_ext.ui.prompts import (
    non_empty_validator,
    prompt_choice,
    prompt_password,
    prompt_text,
)

from ._registry import image_registry_host
from .config import QuickstartConfig
from .container import ContainerManager
from .gpu_config import (
    apply_cuda_visible_devices_filter,
    format_gpu_ids_for_storage,
    parse_cuda_visible_devices_integers,
)

# Create console for formatted output
console = Console()

# Colored symbols using Rich markup
CHECK = "[green]✓[/green]"
CROSS = "[red]✗[/red]"
WARN = "[yellow]![/yellow]"
BULLET = "•"

# Registries that require username/password authentication
REGISTRIES_REQUIRING_AUTH = ("ghcr.io", "docker.io")


@dataclass(frozen=True)
class RegistryCredentials:
    """Validated credentials for an image registry."""

    registry: str
    username: str
    password: SecretStr


def _image_registry_host_from_ref(image: str) -> str:
    # Strip explicit port (e.g. "nvcr.io:443" -> "nvcr.io") for canonical matching.
    return image_registry_host(image).split(":", 1)[0]


def detect_registry_auth_type(image: str) -> Literal["ngc", "user_pass", "none"]:
    """Detect what type of authentication an image registry requires.

    Args:
        image: Container image name (e.g., "nvcr.io/nvidia/nemo:latest")

    Returns:
        "ngc" for nvcr.io (uses NGC API key with $oauthtoken)
        "user_pass" for GitHub Container Registry and Docker Hub
        "none" for other registries (assume already logged in)
    """
    registry_host = _image_registry_host_from_ref(image).lower()

    if registry_host == "nvcr.io":
        return "ngc"

    if registry_host in REGISTRIES_REQUIRING_AUTH:
        return "user_pass"

    return "none"


def prompt_for_configuration(config: QuickstartConfig) -> QuickstartConfig:
    """Interactive configuration wizard using prompt_toolkit.

    Args:
        config: Existing configuration to use as defaults.

    Returns:
        Updated QuickstartConfig with user input.
    """
    requires_registry_auth = detect_registry_auth_type(config.image) == "user_pass"
    total_steps = 4 if requires_registry_auth else 3

    # Header
    console.print("\n[bold]NeMo Helix Quickstart Configuration[/bold]")

    # 1. NGC API Key (required)
    console.print(f"[bold]Step 1 of {total_steps}: NGC Authentication[/bold]")

    current_key = config.ngc_api_key.get_secret_value() if config.ngc_api_key else ""
    if current_key:
        masked_key = f"***{current_key[-4:]}"
        console.print()
        api_key = prompt_password(
            f"NGC API Key [{masked_key}]: ",
        )
    else:
        console.print("Get your NGC API key at: https://org.ngc.nvidia.com/setup/api-key")
        console.print()
        api_key = prompt_password(
            "NGC API Key: ",
            validator=non_empty_validator("NGC API Key"),
        )

    from .validators import validate_ngc_credentials

    normalized_api_key = api_key.strip()
    if normalized_api_key:
        key_to_validate = normalized_api_key
        keep_existing_key = False
    elif current_key:
        key_to_validate = current_key
        keep_existing_key = True
    else:
        key_to_validate = ""
        keep_existing_key = False

    if key_to_validate:
        console.print("Validating NGC API Key...")

        validation = validate_ngc_credentials(key_to_validate)
        if not validation.valid:
            console.print(f"{CROSS} {validation.message}", highlight=False)
            raise RuntimeError(
                "NGC API Key validation failed. Generate a new key at "
                "https://org.ngc.nvidia.com/setup/api-key and try again."
            )
        if keep_existing_key:
            console.print(f"{CHECK} Existing NGC API Key validated")
            console.print("Keeping existing NGC API Key")
        else:
            config.ngc_api_key = SecretStr(key_to_validate)
            console.print(f"{CHECK} NGC API Key validated and saved!")
    else:
        console.print("Keeping existing NGC API Key")

    step_number = 2
    if requires_registry_auth:
        prompt_for_optional_registry_credentials(config)
        step_number += 1

    console.print()
    console.print(f"[bold]Step {step_number} of {total_steps}: GPU Mode[/bold]")

    current_provider = config.inference_provider or "nvidia-build"

    result = prompt_choice(
        message="Select your deployment mode:",
        options=[
            (
                "nvidia-build",
                "Cloud only - Use NVIDIA Build API for inference (no local GPU). Safe Synthesizer is not available.",
            ),
            ("host-gpu", "Host GPU - Use local GPUs for inference and Safe Synthesizer."),
        ],
        default=current_provider,
    )

    config.inference_provider = result  # type: ignore[assignment]
    config.use_gpu = result == "host-gpu"

    if result == "nvidia-build":
        console.print(f"{CHECK} Cloud only selected (NVIDIA Build API)")
    else:
        console.print(f"{CHECK} Host GPU selected")
        # Pre-populate reserved_gpu_device_ids: detect on host, then apply CUDA_VISIBLE_DEVICES filter
        detected = ContainerManager._detect_host_gpu_device_ids()
        if detected:
            filtered = apply_cuda_visible_devices_filter(detected, log_exclusions=True)
            config.reserved_gpu_device_ids = format_gpu_ids_for_storage(filtered)
            console.print(f"{CHECK} Detected {len(filtered)} GPU(s): {config.reserved_gpu_device_ids}")
        else:
            config.reserved_gpu_device_ids = ""
            console.print(
                f"{WARN} No GPUs detected (detection unavailable or no devices). "
                "Set GPU device IDs in advanced options if needed."
            )

    step_number += 1
    console.print()
    console.print(f"[bold]Step {step_number} of {total_steps}: Save Config[/bold]")

    show_advanced_result = prompt_choice(
        message="Save configuration?",
        options=[
            ("yes", "Save configuration"),
            ("no", "Configure advanced options - authentication, ports"),
        ],
        default="yes",
    )
    show_advanced = show_advanced_result == "no"

    if show_advanced:
        # Platform Authorization
        console.print()
        console.print(f"{BULLET} Platform Authorization")
        console.print("Enable auth to require authentication for API requests.")
        console.print("When enabled, you can set an admin email to bootstrap access.")
        console.print()

        auth_result = prompt_choice(
            message="Enable authentication/authorization?",
            options=[
                ("no", "No - Allow all requests without authentication"),
                ("yes", "Yes - Require authentication for API access"),
            ],
            default="yes" if config.auth_enabled else "no",
        )

        config.auth_enabled = auth_result == "yes"

        if config.auth_enabled:
            console.print(f"{CHECK} Authorization enabled")

            # Prompt for admin email
            current_email = config.admin_email or ""
            console.print()
            admin_email = prompt_text(
                f"Admin email (grants HelixAdmin role){f' [{current_email}]' if current_email else ''}: ",
            )

            if admin_email.strip():
                config.admin_email = admin_email.strip()
                console.print(f"{CHECK} Admin: {config.admin_email}")
            elif current_email:
                console.print(f"{CHECK} Keeping existing admin: {current_email}")
            else:
                config.admin_email = None
                console.print(f"{WARN} No admin email set - you'll need to configure access manually")

            # Inform user about CLI authentication behavior
            if config.admin_email:
                console.print()
                console.print(
                    f"[cyan]ℹ[/cyan]  All CLI requests will be authenticated as [bold]{config.admin_email}[/bold]."
                )
                console.print(
                    "   To use a different identity: [cyan]nemo auth login --unsigned-token --email <email>[/cyan]"
                )
        else:
            console.print(f"{CHECK} Authorization disabled - all requests will be allowed")
            config.admin_email = None

        # Host port
        console.print()
        port_input = prompt_text(f"Host port (default: {config.host_port}): ")
        if port_input:
            try:
                port = int(port_input)
                if 1 <= port <= 65535:
                    config.host_port = port
                    console.print(f"{CHECK} Port set to {port}")
                else:
                    console.print(f"{WARN} Invalid port range. Using default.")
            except ValueError:
                console.print(f"{WARN} Invalid port number. Using default.")
        else:
            console.print(f"{CHECK} Using default port {config.host_port}")

        # GPU device IDs (when host-gpu)
        if config.use_gpu:
            console.print()
            console.print(f"{BULLET} GPU device IDs (comma-separated, e.g. 0,1,2)")
            current_gpu = config.reserved_gpu_device_ids or ""
            gpu_input = prompt_text(
                f"GPU device IDs (current: {current_gpu or 'none'}): ",
            )
            if gpu_input is not None and gpu_input.strip():
                parsed = parse_cuda_visible_devices_integers(gpu_input)
                if parsed is None:
                    console.print(f"{WARN} Invalid GPU list (use non-negative integers, e.g. 0,1,2). Keeping current.")
                else:
                    config.reserved_gpu_device_ids = format_gpu_ids_for_storage(parsed)
                    console.print(f"{CHECK} GPU device IDs set to {config.reserved_gpu_device_ids}")
            else:
                console.print(f"{CHECK} Keeping current GPU device IDs: {current_gpu or 'none'}")
    else:
        console.print(f"{CHECK} Using default settings")

    # Summary
    console.print()
    console.print("[bold]Configuration Summary[/bold]")
    provider_display = "Cloud only (NVIDIA Build API)" if config.inference_provider == "nvidia-build" else "Host GPU"
    auth_display = "Enabled" if config.auth_enabled else "Disabled"
    console.print(f"{BULLET} Authorization: {auth_display}")
    if config.auth_enabled and config.admin_email:
        console.print(f"{BULLET} Admin Email: {config.admin_email}")
    console.print(f"{BULLET} GPU Mode: {provider_display}")
    if config.use_gpu and config.reserved_gpu_device_ids:
        console.print(f"{BULLET} GPU device IDs: {config.reserved_gpu_device_ids}")
    console.print(f"{BULLET} Container Image: {config.image}")
    console.print(f"{BULLET} Host Port: {config.host_port}")
    console.print(f"{BULLET} Storage Path: {config.storage_path}", highlight=False)

    return config


def _registry_host_from_image(image: str) -> str:
    """Thin wrapper kept for back-compat with tests that import this name."""
    return image_registry_host(image)


def prompt_for_optional_registry_credentials(config: QuickstartConfig) -> bool:
    """Prompt for private registry credentials during configure when the image needs them.

    Args:
        config: Quickstart configuration whose image determines the registry.

    Returns:
        True when credentials were entered and validated, False when no prompt was needed or the
        user chose to configure them later.
    """
    if detect_registry_auth_type(config.image) != "user_pass":
        return False

    registry = config.get_registry_host() or _registry_host_from_image(config.image)
    registry_label = registry or "the image registry"

    console.print()
    console.print("[bold]Step 2 of 4: Registry Authentication[/bold]")
    console.print(f"{BULLET} {registry_label} requires registry credentials for image pulls.")

    auth_result = prompt_choice(
        message="Configure registry credentials now?",
        options=[
            ("yes", f"Configure credentials for {registry_label}"),
            ("no", "Skip - configure later with nemo quickstart auth"),
        ],
        default="yes",
    )

    if auth_result != "yes":
        console.print(
            f"{WARN} Registry credentials not updated. Run [cyan]nemo quickstart auth[/cyan] "
            "before starting if image pulls fail."
        )
        return False

    credentials = prompt_for_registry_credentials(
        config.image,
        default_registry=registry,
        message=f"Enter a username and token for {registry_label}.",
    )
    config.registry_host = credentials.registry
    config.registry_username = credentials.username
    config.registry_password = credentials.password
    return True


def prompt_for_registry_credentials(
    image: str,
    *,
    default_registry: str | None = None,
    registry: str | None = None,
    username: str | None = None,
    password: str | None = None,
    message: str | None = None,
) -> RegistryCredentials:
    """Collect and validate registry credentials, prompting for missing fields.

    Args:
        image: Container image name (e.g., "registry.example.com/repo/image:tag").
        default_registry: Optional default registry host value.
        registry: Optional registry host value supplied by the caller.
        username: Optional registry username supplied by the caller.
        password: Optional registry password/token supplied by the caller.
        message: Optional message explaining why credentials are being requested.

    Returns:
        Validated registry credentials.
    """
    registry_default = (default_registry or _registry_host_from_image(image)).strip()

    console.print("\n[bold]Image Pull Authentication[/bold]")
    console.print(message or f"Unable to pull {image!r} without credentials.")
    console.print()

    registry_value = (registry or "").strip()
    if not registry_value:
        registry_value = prompt_text(
            "Registry: ",
            default=registry_default,
            validator=non_empty_validator("Registry"),
        ).strip()

    username_value = username.strip() if username else ""
    if not username_value:
        if registry_value.lower().split(":", 1)[0] == "nvcr.io":
            username_value = "$oauthtoken"
        else:
            username_value = prompt_text("Username: ", validator=non_empty_validator("Username")).strip()

    password_value = password.strip() if password is not None else None
    if password_value is None or password_value == "-":
        password_value = prompt_password("Password/Token: ", validator=non_empty_validator("Password")).strip()
    if not password_value:
        raise RuntimeError("Password/token is required")

    console.print("Validating registry credentials...")
    from .validators import validate_registry_credentials

    validation = validate_registry_credentials(registry_value, username_value, password_value)
    if not validation.valid:
        console.print(f"{CROSS} {validation.message}", highlight=False)
        raise RuntimeError(
            f"{validation.message}. Registry credential validation failed for {registry_value}. "
            "Check the username and token, then try again."
        )

    console.print(f"{CHECK} Registry credentials validated")
    return RegistryCredentials(registry=registry_value, username=username_value, password=SecretStr(password_value))
