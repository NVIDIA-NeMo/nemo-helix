# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quickstart CLI commands for managing the NeMo Platform container."""

from __future__ import annotations

import functools
import logging
import re
import threading
import time
from collections import deque
from pathlib import Path

import typer
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from nemo_platform_ext.quickstart.config import QuickstartConfig
from nemo_platform_ext.quickstart.prompts import RegistryCredentials

# Create rich console for formatted output (print to stderr to avoid conflicts with canonical output)
console = Console(stderr=True)
# Console for log stream only (stdout so logs can be piped/redirected)
_console_stdout = Console(stderr=False)

# Configure logger with RichHandler to prevent conflicts with progress bars
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP request logs from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

_DOCKER_REGISTRY_AUTH_STATUS_RE = re.compile(
    r"\b(?:"
    r"(?:http\s+)?(?:status|status\s+code|code)\s*[:=]?\s*(?:401|403)"
    r"|(?:401|403)\s+(?:client|server)\s+error"
    r"|(?:401|403)\s*[:,-]\s*(?:unauthorized|forbidden)"
    r")\b"
)


def handle_errors(func):  # type: ignore[no-untyped-def]
    """Decorator to handle errors in CLI commands."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return func(*args, **kwargs)
        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1)

    return wrapper


def _format_bytes(b: int) -> str:
    """Format bytes as human-readable string."""
    if b >= 1_000_000_000:
        return f"{b / 1_000_000_000:.1f}GB"
    if b >= 1_000_000:
        return f"{b / 1_000_000:.0f}MB"
    if b >= 1_000:
        return f"{b / 1_000:.0f}KB"
    return f"{b}B"


def _check_ready_endpoint(port: int, timeout: float = 2.0) -> bool:
    """Check if the service ready endpoint returns 200.

    Args:
        port: The port the service is running on.
        timeout: Request timeout in seconds.

    Returns:
        True if the ready endpoint returns HTTP 200, False otherwise.
    """
    import httpx

    try:
        response = httpx.get(f"http://localhost:{port}/status", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def _preflight_needs_registry_credentials_prompt(results: list[object]) -> bool:
    """Return True when preflight failed because registry credentials are missing or invalid."""
    promptable_messages = {
        "Registry credentials are required",
        "Registry credentials invalid",
        "Registry credentials invalid or image is inaccessible",
    }
    return any(
        getattr(result, "name", None) == "Registry Credentials"
        and getattr(getattr(result, "status", None), "value", None) == "fail"
        and getattr(result, "message", None) in promptable_messages
        for result in results
    )


def _is_docker_desktop_sign_in_required_error(error: Exception) -> bool:
    """Return True for Docker Desktop org sign-in/proxy authorization failures."""
    message = str(error).lower()
    return (
        "sign in to continue using docker desktop" in message
        or ("proxy authentication required" in message and "docker desktop" in message)
        or "registry.json" in message
    )


def _is_docker_registry_auth_error(error: Exception) -> bool:
    """Return True when Docker reports registry authentication or authorization failure."""
    message = str(error).lower()
    if _DOCKER_REGISTRY_AUTH_STATUS_RE.search(message):
        return True
    return any(
        marker in message
        for marker in (
            "access denied",
            "authentication required",
            "denied: requested access",
            "forbidden",
            "no basic auth credentials",
            "pull access denied",
            "unauthorized",
        )
    )


def _should_prompt_for_pull_credentials(error: Exception, config: QuickstartConfig) -> bool:
    """Return True when an image pull failure can be fixed by username/password registry auth."""
    from nemo_platform_ext.quickstart.prompts import detect_registry_auth_type

    return _is_docker_registry_auth_error(error) and detect_registry_auth_type(config.image) == "user_pass"


def _print_docker_desktop_sign_in_required_error(error: Exception) -> None:
    """Print a targeted recovery hint for Docker Desktop sign-in failures."""
    console.print("\n[red]✗[/red] [red]Docker Desktop sign-in is required to pull this image.[/red]")
    console.print(f"  Error: {error}")
    console.print("  Sign in to Docker Desktop with an account in the required organization, then retry.")


def _configure_registry_auth(
    *,
    image: str | None,
    registry: str | None,
    username: str | None,
    token: str | None,
) -> None:
    """Prompt for registry credentials when needed and validate them with Docker."""
    from nemo_platform_ext.quickstart import prompt_for_registry_credentials
    from nemo_platform_ext.quickstart.prompts import _registry_host_from_image

    if token is not None and token != "-":
        raise RuntimeError("For safety, omit --token or use --token - to prompt securely.")

    config = QuickstartConfig.load()
    target_image = image or config.image or config.resolve_best_image()
    default_registry = _registry_host_from_image(target_image)
    registry_label = (registry or default_registry).strip() or "the image registry"

    console.print("\n[bold]Quickstart Registry Authentication[/bold]")
    if target_image:
        console.print(f"• Image: {target_image}")

    credentials = prompt_for_registry_credentials(
        target_image,
        default_registry=default_registry,
        registry=registry,
        username=username,
        password=token,
        message=f"Enter a username and token for {registry_label}.",
    )
    _save_registry_credentials(config, credentials)
    console.print("• Docker credentials and quickstart registry auth updated.")
    console.print("• You can now run [cyan]nemo quickstart up[/cyan].")


def _save_registry_credentials(config: QuickstartConfig, credentials: RegistryCredentials) -> None:
    """Persist validated registry credentials for the quickstart container jobs backend."""
    config.registry_host = credentials.registry
    config.registry_username = credentials.username
    config.registry_password = credentials.password
    config.save()


class StartupDisplay:
    """Manages the Rich Live display during container startup."""

    def __init__(self) -> None:
        """Initialize startup display."""
        self.status_message = "Initializing..."
        self.log_lines: deque[Text] = deque(maxlen=5)
        self._start_time = time.time()

    def set_status(self, message: str) -> None:
        """Update the status message."""
        self.status_message = message

    def add_log_line(self, line: str) -> None:
        """Add a log line (keeps last 5)."""
        clean_line = line.rstrip("\n")
        if clean_line:
            self.log_lines.append(Text.from_ansi(clean_line))

    def clear_logs(self) -> None:
        """Clear log lines."""
        self.log_lines.clear()

    def render(self) -> Group:
        """Render the current display state."""
        # Spinner + status
        spinner = Spinner("dots")
        status_text = Text()
        status_text.append_text(spinner.render(time.time()))
        status_text.append(f" {self.status_message}")

        components: list[Text | Panel] = [status_text]

        # Add log panel if we have logs
        if self.log_lines:
            # Join Text objects with newlines
            log_content = Text()
            for i, log_line in enumerate(self.log_lines):
                if i > 0:
                    log_content.append("\n")
                log_content.append_text(log_line)
            log_panel = Panel(
                log_content,
                title="[dim]Container Logs[/dim]",
                border_style="dim",
                box=box.HORIZONTALS,
                padding=(0, 1),
            )
            components.append(log_panel)

        return Group(*components)


def _write_quickstart_config(port: int = 8080, *, admin_email: str | None = None) -> None:
    """Write quickstart cluster and context to the NeMo Platform config file.

    Args:
        port: The port the quickstart cluster is running on.
        admin_email: Admin email for authentication when auth is enabled.
    """
    from nemo_platform_ext.config.config import Config
    from nemo_platform_ext.config.models import ConfigParams

    quickstart_context_name = "quickstart"
    config_params: ConfigParams = {
        "base_url": f"http://localhost:{port}",
        "current_context": quickstart_context_name,
        "workspace": "default",
    }
    if admin_email:
        from nemo_platform_ext.auth.helpers import generate_unsigned_jwt

        config_params["access_token"] = generate_unsigned_jwt(
            principal_id=admin_email,
            email=admin_email,
            expires_in_seconds=24 * 60 * 60,
        )

    Config.write(
        config_params,
        context_name=quickstart_context_name,
    )


cluster_info_app = typer.Typer(help="Show information about the connected platform cluster.", add_completion=False)
quickstart_app = typer.Typer(help="Quickstart commands for managing the NeMo Platform container.")


@cluster_info_app.callback(invoke_without_command=True)
def cluster_info(ctx: typer.Context) -> None:
    """Show information about the connected platform cluster."""
    import httpx

    from nemo_platform_ext.cli.version import client_version

    base_url: str | None = None
    context_name: str | None = None
    if ctx.obj is not None:
        base_url = ctx.obj.get_base_url()
        try:
            sdk_context = ctx.obj.get_sdk_context()
            context_name = sdk_context.context_name
        except Exception:
            pass

    if not base_url:
        console.print("[red]✗[/red] No cluster configured.")
        console.print("  Run [cyan]nemo auth login --base-url <URL>[/cyan] to connect to a remote cluster.")
        console.print("  Or run [cyan]nemo quickstart up[/cyan] to start a local quickstart cluster.")
        raise typer.Exit(code=1)

    console.print("\n[bold]Cluster Information[/bold]\n")
    if context_name:
        console.print(f"• Context: {context_name}")
    console.print(f"• URL: {base_url}")
    console.print(f"• CLI Version: {client_version()}")

    try:
        response = httpx.get(f"{base_url.rstrip('/')}/status", timeout=5.0)
    except Exception as e:
        console.print(f"\n[red]✗[/red] Could not reach cluster: {e}")
        raise typer.Exit(code=1)

    if response.status_code != 200:
        console.print(f"\n[red]✗[/red] /status returned HTTP {response.status_code}")
        raise typer.Exit(code=1)

    data = response.json()
    overall = data.get("status", "unknown")
    status_icon = (
        "[green]✓[/green]"
        if overall == "healthy"
        else "[yellow]![/yellow]"
        if overall == "degraded"
        else "[red]✗[/red]"
    )
    console.print(f"{status_icon} Status: {overall}")

    services = data.get("services", {})
    ready = services.get("ready", [])
    not_ready = services.get("not_ready", [])

    if ready:
        console.print(f"\n[green]✓[/green] Ready ({len(ready)}): {', '.join(ready)}")
    for svc in not_ready:
        name = svc.get("name", "unknown") if isinstance(svc, dict) else str(svc)
        msg = svc.get("message", "") if isinstance(svc, dict) else ""
        console.print(f"[red]✗[/red] Not ready: {name}" + (f" — {msg}" if msg else ""))

    controllers = data.get("controllers", {})
    if not controllers.get("healthy", True):
        per_controller = controllers.get("status", {})
        unhealthy = [name for name, ok in per_controller.items() if not ok]
        if unhealthy:
            console.print(f"[red]✗[/red] Unhealthy controllers: {', '.join(unhealthy)}")


@quickstart_app.command()
@handle_errors
def status(ctx: typer.Context) -> None:
    """Show quickstart cluster status and configuration."""
    from nemo_platform_ext.quickstart import QuickstartCluster

    cluster = QuickstartCluster()
    info = cluster.cluster_info()

    if info["status"] == "docker-unavailable":
        console.print("[red]✗[/red] Docker is not running or unavailable.")
        console.print("  quickstart status shows the local quickstart deployment status, which requires Docker.")
        console.print(
            "  Start Docker to check the local quickstart cluster, or run [cyan]nemo quickstart configure[/cyan] to set it up."
        )
        raise typer.Exit(code=1)

    console.print("\n[bold]Quickstart Cluster Information[/bold]\n")

    running_icon = "[green]✓[/green]" if info["running"] else "[red]✗[/red]"
    console.print(f"{running_icon} Running: {'Yes' if info['running'] else 'No'}")
    console.print(f"• Status: {info['status']}")
    console.print(f"• Image: {info['config']['image']}")
    console.print(f"• Port: {info['config']['port']}")
    console.print(f"• Storage: {info['config']['storage_path']}")
    console.print(f"• Docker Socket: {info['config']['docker_socket']}")
    storage_size = info.get("storage_size")
    if storage_size and storage_size not in ["0 B", "0.0 B", "0.00 B"]:
        console.print(f"• Storage Size: {storage_size}")

    if info["running"]:
        health = info.get("health", "unknown")
        health_icon = (
            "[green]✓[/green]"
            if health == "healthy"
            else "[yellow]![/yellow]"
            if health == "starting"
            else "[red]✗[/red]"
        )
        console.print(f"{health_icon} Health: {health}")
        console.print(f"• URL: {info.get('url', 'N/A')}")


@quickstart_app.callback(invoke_without_command=True)
def quickstart_main(ctx: typer.Context) -> None:
    """Quickstart commands for managing the NeMo Platform container."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@quickstart_app.command()
@handle_errors
def configure(
    ctx: typer.Context,
    auto: bool = typer.Option(False, "--auto", help="Auto-configure with defaults (requires NGC_API_KEY env var)"),
) -> None:
    """Configure quickstart settings interactively."""
    from nemo_platform_ext.quickstart import (
        QuickstartConfig,
        prompt_for_configuration,
        validate_config,
        validate_ngc_credentials,
    )

    config = QuickstartConfig.load()

    # Auto mode - use defaults
    if auto:
        console.print("\n")
        console.print("  ╔══════════════════════════════════════════════════════════════╗")
        console.print("  ║                                                              ║")
        console.print("  ║      NeMo Platform Quickstart Auto-Configuration             ║")
        console.print("  ║                                                              ║")
        console.print("  ╚══════════════════════════════════════════════════════════════╝\n")

        if not config.ngc_api_key:
            console.print("  [red]✗[/red] [red]NGC_API_KEY environment variable is not set.[/red]")
            console.print(
                "  • Please set NGC_API_KEY or run [cyan]nemo quickstart configure[/cyan] for interactive setup."
            )
            raise typer.Exit(code=1)

        console.print("  • Validating NGC API Key from environment...")
        validation = validate_ngc_credentials(config.ngc_api_key.get_secret_value())
        if not validation.valid:
            console.print(f"  [red]✗[/red] [red]{validation.message}[/red]", highlight=False)
            console.print(
                "  • Generate a new key at [cyan]https://org.ngc.nvidia.com/setup/api-key[/cyan] "
                "and update [cyan]NGC_API_KEY[/cyan]."
            )
            raise typer.Exit(code=1)

        console.print("  [green]✓[/green] NGC API Key validated\n")
        console.print("  [bold]Configuration:[/bold]")
        console.print(f"    • Image:   {config.image}")
        console.print(f"    • Port:    {config.host_port}")
        console.print(f"    • Storage: {config.storage_path}")

        config.save()
        console.print(
            f"\n  [green]✓[/green] [green]Configuration saved to {config.get_default_config_path()}[/green]\n"
        )

        # Inform user about authentication when auth is enabled
        if config.auth_enabled and config.admin_email:
            console.print(
                f"  [cyan]ℹ[/cyan]  Authentication enabled: CLI requests will be authenticated as "
                f"[bold]{config.admin_email}[/bold] (platform admin)."
            )
            console.print(
                "      To use a different identity after starting, run: "
                "[cyan]nemo auth login --unsigned-token --email <email>[/cyan]\n"
            )
        return

    # Interactive mode
    config = prompt_for_configuration(config)

    # Validate
    console.print()
    console.print("• Validating configuration...")
    console.print()
    results = validate_config(config)

    all_valid = True
    for result in results:
        if result.valid:
            console.print(f"[green]✓[/green] {result.message}", highlight=False)
        else:
            console.print(f"[red]✗[/red] {result.message}", highlight=False)
            all_valid = False

    if not all_valid:
        console.print()
        console.print("[red]✗[/red] [red]Configuration validation failed. Please fix the issues above.[/red]")
        raise typer.Exit(1)

    config.save()
    console.print()
    console.print("[green]✓[/green] [green bold]Configuration saved successfully![/green bold]")
    console.print(f"• {config.get_default_config_path()}", highlight=False)

    console.print()
    console.print("[bold cyan]Next steps:[/bold cyan]")
    console.print("• Run [cyan]nemo quickstart up[/cyan] to start the cluster")


@quickstart_app.command("auth")
@handle_errors
def auth(
    _ctx: typer.Context,
    image: str | None = typer.Option(
        None,
        "--image",
        help="Container image whose registry should be authenticated.",
    ),
    registry: str | None = typer.Option(
        None,
        "--registry",
        help="Registry host to authenticate, for example ghcr.io.",
    ),
    username: str | None = typer.Option(
        None,
        "--username",
        "-u",
        help="Registry username. For nvcr.io, defaults to $oauthtoken.",
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        "--password",
        help="Use '-' or omit this option to prompt securely for the registry password/token. Inline values are rejected.",
        hide_input=True,
    ),
) -> None:
    """Configure and validate registry credentials for quickstart image pulls."""
    _configure_registry_auth(image=image, registry=registry, username=username, token=token)


@quickstart_app.command()
@handle_errors
def up(
    ctx: typer.Context,
    config_path: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to platform configuration YAML file",
    ),
    image: str | None = typer.Option(
        None,
        "--image",
        help="Container image to use (overrides default image)",
    ),
    skip_preflight: bool = typer.Option(
        False,
        "--skip-preflight",
        help="Skip pre-flight checks",
    ),
    no_pull: bool = typer.Option(
        False,
        "--no-pull",
        help="Don't pull the container image",
    ),
    timeout: int = typer.Option(
        300,
        "--timeout",
        "-t",
        help="Maximum time to wait for service to become healthy (seconds)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Restart the service if already running",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Accept all prompts with their defaults",
    ),
) -> None:
    """Start the quickstart cluster."""
    from nemo_platform_ext.quickstart import (
        CheckStatus,
        QuickstartCluster,
        QuickstartConfig,
        is_interactive,
        prompt_for_registry_credentials,
    )

    console.print("\n[bold cyan]Starting NeMo Platform Quickstart[/bold cyan]\n")

    quickstart_config = QuickstartConfig.load()

    # Require reserved_gpu_device_ids when host-gpu is enabled (no backwards compatibility)
    if quickstart_config.use_gpu:
        gpu_ids = quickstart_config.reserved_gpu_device_ids
        if not gpu_ids or not gpu_ids.strip():
            console.print(
                "[red]GPU mode is enabled but GPU device IDs are not set.[/red]\n"
                "Re-run [cyan]nemo quickstart configure[/cyan] and select host-gpu to detect and set GPU device IDs,"
                "or use advanced options to choose which GPUs to use."
            )
            raise typer.Exit(1)

    # Override image if provided
    if image:
        quickstart_config.image = image
    else:
        # Auto-detect internal nightly image if the NGC key has access
        resolved = quickstart_config.resolve_best_image()
        if resolved and resolved != quickstart_config.image:
            console.print(f"[dim]Using nightly image: {resolved}[/dim]\n")
            quickstart_config.image = resolved
        elif not quickstart_config.image and not quickstart_config.ngc_api_key:
            console.print("[yellow]! No NGC API key configured — cannot auto-select nightly image.[/yellow]")
            console.print("[yellow]  Run 'nemo quickstart init' to configure credentials, or pass --image.[/yellow]\n")

    cluster = QuickstartCluster(
        config=quickstart_config,
        platform_config_path=config_path,
    )

    # Run pre-flight checks
    if not skip_preflight:
        console.print("Running pre-flight checks...\n")
        results = cluster.preflight()

        for result in results:
            icon = {
                CheckStatus.PASS: "[green]✓[/green]",
                CheckStatus.WARN: "[yellow]![/yellow]",
                CheckStatus.FAIL: "[red]✗[/red]",
            }[result.status]
            console.print(f"{icon} {result.name}: {result.message}")

        if cluster._preflight_checker.has_failures():
            if not yes and is_interactive() and _preflight_needs_registry_credentials_prompt(results):
                console.print("\n[yellow]! Registry credentials are required to access this image.[/yellow]")
                registry_host_default = quickstart_config.get_registry_host()
                credentials = prompt_for_registry_credentials(
                    quickstart_config.image,
                    default_registry=registry_host_default,
                )
                _save_registry_credentials(quickstart_config, credentials)

                console.print("\nRe-running pre-flight checks...\n")
                results = cluster.preflight()
                for result in results:
                    icon = {
                        CheckStatus.PASS: "[green]✓[/green]",
                        CheckStatus.WARN: "[yellow]![/yellow]",
                        CheckStatus.FAIL: "[red]✗[/red]",
                    }[result.status]
                    console.print(f"{icon} {result.name}: {result.message}")

        if cluster._preflight_checker.has_failures():
            console.print("\n[red]✗[/red] [red]Pre-flight checks failed. Fix issues and retry.[/red]")
            raise typer.Exit(1)

        if cluster._preflight_checker.is_already_running():
            # Force restart if --image is passed or --force is used
            if force or image:
                reason = "with new image" if image else ""
                console.print(f"\n• [yellow]Cluster is already running. Restarting {reason}...[/yellow]")
                cluster.stop()
                console.print("[green]✓[/green] Cluster stopped.\n")
            else:
                console.print("\n[green]✓[/green] [green]NeMo Platform Quickstart is already running![/green]")
                status = cluster.status()
                if url := status.get("url"):
                    console.print(f"• URL: {url}")
                return

        if cluster._preflight_checker.has_warnings():
            console.print("\n[yellow]![/yellow]  [yellow]Some checks have warnings. Proceeding anyway...[/yellow]")

    # Create display manager for progress UI
    display = StartupDisplay()
    final_health = "unknown"

    try:
        with Live(display.render(), console=console, refresh_per_second=10) as live:
            # Phase 1: Pull image (if needed)
            if not no_pull:
                display.set_status("Pulling image...")
                live.update(display.render())

                # Track progress per layer for overall percentage
                layer_progress: dict[str, dict[str, int]] = {}

                def pull_image_with_updates(registry_auth: dict[str, str] | None = None) -> None:
                    for progress in cluster._container_manager.pull_image_with_progress(auth_override=registry_auth):
                        layer_id = progress.get("layer_id")
                        current = progress.get("current")
                        total = progress.get("total")

                        # Update layer tracking
                        if layer_id and total:
                            layer_progress[layer_id] = {"current": current or 0, "total": total}

                        # Calculate overall percentage
                        total_bytes = sum(lp["total"] for lp in layer_progress.values())
                        current_bytes = sum(lp["current"] for lp in layer_progress.values())

                        if total_bytes > 0:
                            percent = int(current_bytes / total_bytes * 100)
                            display.set_status(
                                f"Pulling image... {percent}% ({_format_bytes(current_bytes)} / {_format_bytes(total_bytes)})"
                            )
                        else:
                            status_text = progress.get("status", "")
                            display.set_status(f"Pulling image... {status_text}")
                        live.update(display.render())

                try:
                    pull_image_with_updates()
                except Exception as pull_error:
                    if _is_docker_desktop_sign_in_required_error(pull_error):
                        live.stop()
                        _print_docker_desktop_sign_in_required_error(pull_error)
                        raise typer.Exit(1)

                    if (
                        yes
                        or not is_interactive()
                        or not _should_prompt_for_pull_credentials(pull_error, quickstart_config)
                    ):
                        raise

                    live.stop()
                    console.print("\n[yellow]!Unable to pull image.[/yellow]")
                    console.print(f"  Error: {pull_error}")

                    registry_host_default = quickstart_config.get_registry_host()
                    credentials = prompt_for_registry_credentials(
                        quickstart_config.image,
                        default_registry=registry_host_default,
                    )
                    _save_registry_credentials(quickstart_config, credentials)
                    live.start()
                    display.set_status("Pulling image...")
                    live.update(display.render())
                    try:
                        pull_image_with_updates(
                            {
                                "registry": credentials.registry,
                                "username": credentials.username,
                                "password": credentials.password.get_secret_value(),
                            }
                        )
                    except Exception as retry_pull_error:
                        if _is_docker_desktop_sign_in_required_error(retry_pull_error):
                            live.stop()
                            _print_docker_desktop_sign_in_required_error(retry_pull_error)
                            raise typer.Exit(1)
                        raise

            # Phase 2: Start container
            display.set_status("Starting container...")
            display.clear_logs()
            live.update(display.render())

            cluster._container_manager.start(
                platform_config=cluster.platform_config,
                pull=False,  # Already pulled above
            )

            # Phase 3: Wait for healthy with log streaming
            display.set_status("Waiting for service to become healthy...")
            live.update(display.render())

            # Start log streaming in background thread
            stop_logs = threading.Event()

            def stream_logs() -> None:
                try:
                    for line in cluster.logs(follow=True, tail=5):
                        if stop_logs.is_set():
                            break
                        display.add_log_line(line)
                        live.update(display.render())
                except Exception:
                    pass  # Container may stop or not exist yet

            log_thread = threading.Thread(target=stream_logs, daemon=True)
            log_thread.start()

            # Poll for ready endpoint
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Check if container is still running
                status = cluster.status()
                if not status.get("running"):
                    final_health = "stopped"
                    stop_logs.set()
                    break

                # Check ready endpoint
                if _check_ready_endpoint(quickstart_config.host_port):
                    final_health = "healthy"
                    stop_logs.set()
                    break

                display.set_status("Waiting for service to be ready...")
                live.update(display.render())
                time.sleep(2)
            else:
                # Timeout reached
                stop_logs.set()

            stop_logs.set()
            log_thread.join(timeout=1)

        # After Live context - display final result
        if final_health == "healthy":
            status = cluster.status()
            console.print(f"\n[green]✓[/green] [green]Quickstart cluster is healthy at {status['url']}[/green]")

            # Write quickstart context to config file
            try:
                # Pass admin_email for authentication when auth is enabled
                config_admin_email = quickstart_config.admin_email if quickstart_config.auth_enabled else None
                _write_quickstart_config(port=quickstart_config.host_port, admin_email=config_admin_email)
                console.print("[green]✓[/green] Context 'quickstart' added to config")

                # Inform user about authentication when auth is enabled
                if quickstart_config.auth_enabled and quickstart_config.admin_email:
                    console.print()
                    console.print(
                        f"[cyan]ℹ[/cyan]  Authentication enabled: All CLI requests will be authenticated as "
                        f"[bold]{quickstart_config.admin_email}[/bold] (platform admin)."
                    )
                    console.print(
                        "    To use a different identity, run: "
                        "[cyan]nemo auth login --unsigned-token --email <email>[/cyan]"
                    )
            except Exception as e:
                console.print(f"[yellow]![/yellow]  [yellow]Warning: Could not write config: {e}[/yellow]")

        elif final_health == "stopped":
            console.print("\n[red]✗[/red] [red]Container stopped unexpectedly![/red]")
            console.print("[red]Error logs:[/red]\n")
            for line in cluster.logs(follow=False, tail=100):
                console.print(Text.from_ansi(line), end="")
            raise typer.Exit(1)

        else:
            console.print("\n[red]✗[/red] [red]Timeout waiting for service to be ready[/red]")
            console.print("[yellow]Recent logs:[/yellow]\n")
            for line in cluster.logs(follow=False, tail=50):
                console.print(Text.from_ansi(line), end="")
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[red]✗[/red] [red]Failed to start cluster: {e}[/red]")
        # Try to get logs if container exists
        try:
            for line in cluster.logs(follow=False, tail=50):
                console.print(Text.from_ansi(line), end="")
        except Exception:
            pass
        raise typer.Exit(1)

    console.print()
    console.print("[bold cyan]Next steps:[/bold cyan]")
    console.print("• Run [cyan]nemo quickstart status[/cyan] to check cluster status")
    console.print("• Run [cyan]nemo quickstart logs -f[/cyan] to follow logs")
    if quickstart_config.inference_provider == "nvidia-build":
        console.print("• Chat with Nemotron: [cyan]nemo chat nvidia-llama-3-3-nemotron-super-49b-v1-5[/cyan]")


@quickstart_app.command()
@handle_errors
def down(ctx: typer.Context) -> None:
    """Stop the quickstart cluster."""
    from nemo_platform_ext.quickstart import QuickstartCluster

    cluster = QuickstartCluster()

    status = cluster.status()
    if not status["running"]:
        console.print("Cluster is not running.")
        return

    console.print("• Stopping quickstart cluster...")
    cluster.stop()
    console.print("[green]✓[/green] Cluster stopped.")


@quickstart_app.command()
@handle_errors
def destroy(
    ctx: typer.Context,
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        "-f",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Stop the cluster and remove all data and configuration files."""
    from nemo_platform_ext.quickstart import QuickstartCluster, QuickstartConfig, prompt_confirm

    cluster = QuickstartCluster()

    console.print("[yellow]![/yellow]  [yellow]This will remove all quickstart data and configuration![/yellow]")
    if not yes and not prompt_confirm("Are you sure?"):
        console.print("• Cancelled.")
        raise typer.Exit(0)

    console.print("•  Destroying quickstart cluster...")
    cluster.destroy()
    console.print("[green]✓[/green] Cluster destroyed and data removed.")

    console.print("•  Removing quickstart configuration...")
    QuickstartConfig.remove()
    console.print("[green]✓[/green] Configuration removed.")


@quickstart_app.command()
@handle_errors
def logs(
    ctx: typer.Context,
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    tail: int = typer.Option(100, "--tail", "-n", help="Number of lines to show from the end"),
    all_logs: bool = typer.Option(False, "--all", "-a", help="Show all logs (overrides --tail)"),
) -> None:
    """View cluster logs."""
    from nemo_platform_ext.quickstart import QuickstartCluster

    cluster = QuickstartCluster()

    status = cluster.status()
    if status["status"] == "not found":
        console.print("[red]✗[/red] No container found. Run [cyan]nemo quickstart up[/cyan] first.")
        raise typer.Exit(1)

    if not status["running"]:
        console.print(
            f"[yellow]![/yellow]  [yellow]Container is {status['status']}. Showing available logs:[/yellow]\n"
        )
        if follow:
            console.print("•  [yellow]Note: --follow has no effect on stopped containers.[/yellow]\n")
        follow = False  # Can't follow a stopped container

    effective_tail = None if all_logs else tail
    for line in cluster.logs(follow=follow, tail=effective_tail):
        # Strip trailing newline if present; print to stdout so logs can be piped/redirected
        _console_stdout.print(Text.from_ansi(line.rstrip("\n")))


def _run_job_diagnostic(port: int, registry: str, tag: str, *, admin_email: str | None = None) -> bool:
    """Run a diagnostic job to verify the job system is working correctly.

    Args:
        port: The port the NeMo Platform service is running on.
        registry: The image registry to use.
        tag: The image tag to use.
        admin_email: Admin email for authentication when auth is enabled.

    Returns:
        True if the job completed successfully, False otherwise.
    """
    import uuid

    from nemo_platform_plugin.jobs.client import JobsClient
    from nemo_platform_plugin.jobs.types import CreatePlatformJobRequest

    # When auth is enabled, use an unsigned JWT for the admin principal.
    default_headers = None
    if admin_email:
        from nemo_platform_ext.auth.helpers import generate_unsigned_jwt

        default_headers = {
            "Authorization": "Bearer "
            + generate_unsigned_jwt(
                principal_id=admin_email,
                email=admin_email,
                expires_in_seconds=24 * 60 * 60,
            )
        }

    try:
        jobs_client = JobsClient(
            base_url=f"http://localhost:{port}",
            workspace="default",
            default_headers=default_headers,
        )

        # Construct the CPU task image name from the registry and tag
        cpu_image = f"{registry}/nmp-cpu-tasks:{tag}"
        console.print(f"  • Task image: {cpu_image}")

        # Create a job that runs a simple diagnostic command
        job_name = f"diagnostic-{uuid.uuid4().hex[:8]}"
        console.print(f"  • Creating diagnostic job: {job_name}")

        job = jobs_client.create_job(
            body=CreatePlatformJobRequest(
                platform_spec={
                    "steps": [
                        {
                            "name": "diagnostic",
                            "executor": {
                                "provider": "cpu",
                                "container": {
                                    "image": cpu_image,
                                    "entrypoint": [
                                        "python",
                                        "-c",
                                        "import sys; print(f'Python {sys.version}'); print('Job system is working correctly!')",
                                    ],
                                },
                            },
                        }
                    ]
                },
                source="quickstart-doctor",
                spec={},
                name=job_name,
            )
        ).data()

        console.print("  • Waiting for job to complete...")

        # Poll for job completion (max 60 seconds)
        max_wait = 60
        poll_interval = 2
        elapsed = 0
        status = "pending"
        job_status = jobs_client.get_job(name=job.name).data()

        while elapsed < max_wait:
            job_status = jobs_client.get_job(name=job.name).data()
            status = job_status.status

            if status in ("completed", "error", "cancelled"):
                break

            time.sleep(poll_interval)
            elapsed += poll_interval

        success = False
        if status == "completed":
            console.print("  • Diagnostic job completed successfully")
            success = True
        elif status == "error":
            console.print("  • [red]Diagnostic job failed[/red]")
            # Show error details if available
            if hasattr(job_status, "error_details") and job_status.error_details:
                error_msg = job_status.error_details.get("message", str(job_status.error_details))
                console.print(f"    Error: {error_msg}")
        elif status == "cancelled":
            console.print("  • [yellow]Diagnostic job was cancelled[/yellow]")
        else:
            console.print(f"  • [yellow]Job timed out (status: {status})[/yellow]")

        # Fetch and display logs
        console.print("\n  [bold]Job output:[/bold]")
        try:
            logs = jobs_client.list_job_logs(name=job.name)
            log_lines = []
            for log_entry in logs.items():
                if hasattr(log_entry, "message"):
                    log_lines.append(log_entry.message)

            if log_lines:
                for line in log_lines:
                    console.print(f"    {line}")
            else:
                console.print("    [dim](no logs available)[/dim]")
        except Exception as e:
            console.print(f"    Could not fetch logs: {e}")

        # Clean up the job (only if successful)
        if status == "completed":
            try:
                jobs_client.delete_job(name=job.name)
            except Exception:
                pass  # Ignore cleanup errors
        else:
            console.print(f"\n  Job '{job_name}' kept for inspection. Delete with: nemo api jobs delete {job_name}")

        return success

    except Exception as e:
        console.print(f"  • [red]Error running job diagnostic: {e}[/red]")
        return False


@quickstart_app.command()
@handle_errors
def doctor(ctx: typer.Context) -> None:
    """Diagnose quickstart configuration and image settings."""
    from nemo_platform_ext.quickstart import QuickstartCluster, QuickstartConfig

    console.print("\n[bold cyan]NeMo Platform Quickstart Doctor[/bold cyan]\n")

    # Load and display quickstart config
    config = QuickstartConfig.load()
    console.print("[bold]Quickstart Configuration:[/bold]")
    console.print(f"  • Image: {config.image}")

    # Parse and display image components
    registry, tag = config.parse_image_components()
    console.print(f"  • Parsed Registry: {registry or '(empty)'}")
    console.print(f"  • Parsed Tag: {tag}")

    # Check if container is running
    cluster = QuickstartCluster(config=config)
    status = cluster.status()

    if not status["running"]:
        console.print(
            "\n[yellow]Container is not running. Start it with 'nemo quickstart up' to see runtime config.[/yellow]"
        )
        return

    console.print("\n[bold]Container Environment:[/bold]")

    # Get relevant environment variables from the container
    container = cluster._container_manager._get_container()
    if container:
        # Execute a command inside the container to get the platform config
        try:
            # Get the image registry/tag env vars
            exit_code, output = container.exec_run(
                [
                    "sh",
                    "-c",
                    "echo NMP_IMAGE_REGISTRY=$NMP_IMAGE_REGISTRY && echo NMP_IMAGE_TAG=$NMP_IMAGE_TAG",
                ]
            )
            if exit_code == 0:
                for line in output.decode().strip().split("\n"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        console.print(f"  • {key}: {value or '(not set)'}")
        except Exception as e:
            console.print(f"  • Could not read env vars: {e}")

        # Get resolved platform config from inside the container
        console.print("\n[bold]Resolved Platform Config:[/bold]")
        try:
            exit_code, output = container.exec_run(
                [
                    "python",
                    "-c",
                    """
from nmp.common.config import get_platform_config
config = get_platform_config()
print(f"image_registry: {config.image_registry}")
print(f"image_tag: {config.image_tag}")
""",
                ]
            )
            if exit_code == 0:
                for line in output.decode().strip().split("\n"):
                    console.print(f"  • {line}")
            else:
                console.print(f"  • [red]Error: {output.decode()}[/red]")
        except Exception as e:
            console.print(f"  • Could not read platform config: {e}")

        # Run a diagnostic job to verify the job system is working
        console.print("\n[bold]Job System Diagnostics:[/bold]")
        job_success = False
        if registry:
            admin_email = config.admin_email if config.auth_enabled else None
            job_success = _run_job_diagnostic(config.host_port, registry, tag, admin_email=admin_email)
        else:
            console.print("  • [yellow]No registry configured. Job diagnostics skipped.[/yellow]")

        # Final status
        if job_success:
            console.print("\n[green]✓[/green] Quickstart diagnostics passed")
        else:
            console.print("\n[red]✗[/red] Quickstart diagnostics failed")
