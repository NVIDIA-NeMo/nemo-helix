# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Emoji constants for different message types
from rich.console import Console

EMOJI = {
    "success": "✅",
    "error": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
    "rocket": "🚀",
    "package": "📦",
    "wrench": "🔧",
    "hammer": "🔨",
    "check": "✓",
    "cross": "✗",
    "hourglass": "⏳",
    "sparkles": "✨",
    "fire": "🔥",
    "link": "🔗",
    "key": "🔑",
    "lock": "🔒",
    "unlock": "🔓",
    "cloud": "☁️",
    "server": "🖥️",
    "network": "🌐",
    "arrow_right": "→",
    "bullet": "•",
    "save": "💾",
    "auth": "🔐",
    "skip": "⏭️",
    "workspace": "📁",
    "preferences": "⚙️",
}

# Color scheme constants
COLORS = {
    "success": "green",
    "error": "red",
    "warning": "yellow",
    "info": "cyan",
    "dim": "bright_black",
    "primary": "blue",
    "secondary": "magenta",
    "highlight": "bright_cyan",
}

# Pre-configured console instances for all CLI output
console = Console()
console_err = Console(stderr=True)


def success(message: str, emoji: bool = True) -> None:
    """Print a success message in green with optional emoji."""
    prefix = f"{EMOJI['success']} " if emoji else ""
    console.print(f"{prefix}[bold {COLORS['success']}]{message}[/]")


def error(message: str, emoji: bool = True) -> None:
    """Print an error message in red with optional emoji to stderr."""
    prefix = f"{EMOJI['error']} " if emoji else ""
    console_err.print(f"{prefix}[bold {COLORS['error']}]{message}[/]", style="red")


def warning(message: str, emoji: bool = True) -> None:
    """Print a warning message in yellow with optional emoji to stderr."""
    prefix = f"{EMOJI['warning']} " if emoji else ""
    console_err.print(f"{prefix}[bold {COLORS['warning']}]{message}[/]")


def info(message: str, emoji: bool = True) -> None:
    """Print an info message in cyan with optional emoji."""
    prefix = f"{EMOJI['info']} " if emoji else ""
    console.print(f"{prefix}[{COLORS['info']}]{message}[/]")


def section(title: str) -> None:
    """Print a section header with decorative styling."""
    console.print(f"\n[bold {COLORS['primary']}]━━━ {title} ━━━[/]\n")


def subsection(title: str) -> None:
    """Print a subsection header."""
    console.print(f"[bold {COLORS['secondary']}]{title}[/]")


def status(message: str, emoji_key: str | None = None) -> None:
    """Print a status message with optional emoji."""
    prefix = f"{EMOJI.get(emoji_key, '')} " if emoji_key else ""
    console.print(f"{prefix}{message}")


def key_value(key: str, value: str, indent: int = 0) -> None:
    """Print a key-value pair with consistent formatting."""
    spacing = " " * indent
    console.print(f"{spacing}[bold]{key}:[/] [{COLORS['highlight']}]{value}[/]")


def bullet_list(items: list[str], indent: int = 0) -> None:
    """Print a bulleted list of items."""
    spacing = " " * indent
    for item in items:
        console.print(f"{spacing}{EMOJI['bullet']} {item}")


def divider() -> None:
    """Print a visual divider line."""
    console.print(f"[{COLORS['dim']}]{'─' * console.width}[/]")
