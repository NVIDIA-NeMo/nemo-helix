# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive prompts for CLI and SDK applications.

This module provides unified prompt functions built on prompt_toolkit for
text input, password input, confirmation, and selection prompts.

All prompt functions raise `UserCancelled` if the user cancels (Ctrl+C/EOF).
Callers can catch this once at the top level to handle cancellation cleanly.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, TypeVar, overload

from nemo_helix_plugin.entity_naming import NAME_PATTERN, NAME_PATTERN_DESCRIPTION
from prompt_toolkit import PromptSession, prompt
from prompt_toolkit.application import Application
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import ValidationError, Validator

from nemo_helix_ext.ui.output import console

T = TypeVar("T")


class UserCancelled(Exception):
    """Raised when the user cancels a prompt (Ctrl+C or EOF)."""


# Style for prompts: bold cyan prompt text, default user input
PROMPT_STYLE = Style.from_dict(
    {
        "prompt": "ansicyan bold",  # Standard terminal ANSI cyan, bold
        "bottom-toolbar": "noreverse ansiblue italic",  # Hint in bottom toolbar - dim
        "": "",  # Default (user input) - no color override
    }
)


def _make_bottom_toolbar(hint: str | None):
    """Create bottom toolbar with hint."""
    if not hint:
        return None
    return HTML(f"  Hint: <bottom-toolbar>{hint}</bottom-toolbar>")


def _print_confirmation(
    confirmation: str | Callable[[T], str],
    result: T,
    indent: int,
) -> None:
    """Print confirmation message after successful input.

    Supports rich markup in the confirmation message.
    """
    prefix = " " * indent
    if callable(confirmation):
        msg = confirmation(result)
    else:
        msg = confirmation
    console.print(f"[dim]{prefix}{msg}[/]\n")


def is_interactive() -> bool:
    """Check if the session is interactive (can prompt for input).

    Returns:
        True if stdin is a TTY (terminal), False otherwise (CI/piped input).
    """
    return sys.stdin.isatty()


class NonEmptyValidator(Validator):
    """Validates that input is not empty."""

    def __init__(self, field_name: str = "This field"):
        """Initialize validator.

        Args:
            field_name: Name of the field for error messages.
        """
        self.field_name = field_name

    def validate(self, document: Document) -> None:
        """Validate that the input is not empty.

        Args:
            document: The prompt_toolkit document containing user input.

        Raises:
            ValidationError: If the input is empty.
        """
        if not document.text.strip():
            raise ValidationError(
                message=f"{self.field_name} cannot be empty",
                cursor_position=0,
            )


def non_empty_validator(name: str) -> Validator:
    """Create a validator that ensures input is not empty.

    Args:
        name: The field name to use in error messages.

    Returns:
        A Validator instance.
    """
    return NonEmptyValidator(name)


_PROVIDER_NAME_RE = re.compile(NAME_PATTERN)


class ProviderNameValidator(Validator):
    """Validates that input is a legal provider name.

    Uses the entity-store ``NAME_PATTERN`` (RFC 1035-like) so the user gets
    instant feedback instead of a 422 after submission.
    """

    def validate(self, document: Document) -> None:
        text = document.text
        if not text.strip() or not _PROVIDER_NAME_RE.fullmatch(text):
            raise ValidationError(message=NAME_PATTERN_DESCRIPTION, cursor_position=0)


def provider_name_validator() -> Validator:
    """Create a validator for provider names."""
    return ProviderNameValidator()


def prompt_text(
    message: str,
    *,
    default: str = "",
    validator: Validator | None = None,
    hint: str | None = None,
    indent: int = 0,
    confirmation: str | Callable[[str], str] | None = None,
) -> str:
    """Prompt for text input.

    Args:
        message: The prompt message to display.
        default: Default value if user presses Enter.
        validator: Optional validator for the input.
        hint: Optional hint text to display beside the prompt.
        indent: Number of spaces to indent the prompt.
        confirmation: Optional confirmation message. Can be a static string or a callable that receives the result and returns a message.

    Returns:
        The user's input.

    Raises:
        UserCancelled: If the user cancels (Ctrl+C/EOF).
    """
    prefix = " " * indent
    try:
        result = prompt(
            HTML(f"{prefix}<prompt>{message}</prompt>"),
            default=default,
            validator=validator,
            style=PROMPT_STYLE,
            bottom_toolbar=_make_bottom_toolbar(hint),
        )
    except (KeyboardInterrupt, EOFError):
        raise UserCancelled from None

    if confirmation is not None:
        _print_confirmation(confirmation, result, indent)
    return result


def prompt_password(
    message: str,
    *,
    validator: Validator | None = None,
    hint: str | None = None,
    indent: int = 0,
    confirmation: str | Callable[[str], str] | None = None,
) -> str:
    """Prompt for password input (hidden).

    Args:
        message: The prompt message to display.
        validator: Optional validator for the input.
        hint: Optional hint text to display beside the prompt.
        indent: Number of spaces to indent the prompt.
        confirmation: Optional confirmation message. Can be a static string or
            a callable that receives the result and returns a message.

    Returns:
        The user's input.

    Raises:
        UserCancelled: If the user cancels (Ctrl+C/EOF).
    """
    prefix = " " * indent
    try:
        result = prompt(
            HTML(f"{prefix}<prompt>{message}</prompt>"),
            is_password=True,
            validator=validator,
            style=PROMPT_STYLE,
            bottom_toolbar=_make_bottom_toolbar(hint),
        )
    except (KeyboardInterrupt, EOFError):
        raise UserCancelled from None

    if confirmation is not None:
        _print_confirmation(confirmation, result, indent)
    return result


def prompt_confirm(
    message: str,
    *,
    default: bool = False,
    hint: str | None = None,
    indent: int = 0,
    confirmation: str | Callable[[bool], str] | None = None,
) -> bool:
    """Prompt for yes/no confirmation.

    Accepts 'y' or 'n' immediately without requiring Enter.
    Press Enter alone to accept the default value.

    Args:
        message: The confirmation message to display.
        default: Default value if user presses Enter.
        hint: Optional hint text to display below the prompt.
        indent: Number of spaces to indent the prompt.
        confirmation: Optional confirmation message. Can be a static string or
            a callable that receives the result (bool) and returns a message.

    Returns:
        True if user confirmed, False if declined.

    Raises:
        UserCancelled: If the user cancels (Ctrl+C/EOF).
    """
    bindings = KeyBindings()

    @bindings.add("y")
    @bindings.add("Y")
    def accept_yes(event: KeyPressEvent) -> None:
        event.current_buffer.text = "y"
        event.current_buffer.validate_and_handle()

    @bindings.add("n")
    @bindings.add("N")
    def accept_no(event: KeyPressEvent) -> None:
        event.current_buffer.text = "n"
        event.current_buffer.validate_and_handle()

    prefix = " " * indent
    suffix = " [Y/n]: " if default else " [y/N]: "
    session: PromptSession[str] = PromptSession(key_bindings=bindings, style=PROMPT_STYLE)

    try:
        response = (
            session.prompt(
                HTML(f"{prefix}<prompt>{message}{suffix}</prompt>"),
                bottom_toolbar=_make_bottom_toolbar(hint),
            )
            .strip()
            .lower()
        )
    except (KeyboardInterrupt, EOFError):
        raise UserCancelled from None

    result = default if not response else response.startswith("y")
    if confirmation is not None:
        _print_confirmation(confirmation, result, indent)
    return result


def prompt_select(
    message: str,
    choices: Sequence[str] | Sequence[tuple[str, str]],
    *,
    default: str | None = None,
    hint: str | None = None,
    indent: int = 0,
    confirmation: str | Callable[[str], str] | None = None,
) -> str:
    """Prompt user to select from a list of options.

    Args:
        message: The prompt message.
        choices: Sequence of options. Can be plain strings or (value, label) tuples.
            If tuples, the value is returned and the label is displayed.
        default: Default value if user presses Enter.
        hint: Optional hint text to display below the prompt.
        indent: Number of spaces to indent the prompt.
        confirmation: Optional confirmation message. Can be a static string or
            a callable that receives the result and returns a message.

    Returns:
        Selected value string.

    Raises:
        UserCancelled: If the user cancels (Ctrl+C/EOF).
    """
    # Normalize to (value, label) tuples
    normalized: list[tuple[str, str]] = []
    for choice in choices:
        if isinstance(choice, tuple):
            normalized.append(choice)
        else:
            normalized.append((choice, choice))

    # Find default index
    default_idx = None
    for i, (value, _) in enumerate(normalized):
        if value == default:
            default_idx = i + 1
            break

    prefix = " " * indent
    print(f"{prefix}{message}")
    for i, (value, label) in enumerate(normalized, 1):
        marker = " (default)" if value == default else ""
        print(f"{prefix}{i}. {label}{marker}")

    while True:
        try:
            default_str = str(default_idx) if default_idx else ""
            response = prompt(
                HTML(f"{prefix}<prompt>Select [1-{len(normalized)}]: </prompt>"),
                default=default_str,
                style=PROMPT_STYLE,
                bottom_toolbar=_make_bottom_toolbar(hint),
            ).strip()
        except (KeyboardInterrupt, EOFError):
            raise UserCancelled from None

        # Check if it's a number
        try:
            idx = int(response)
            if 1 <= idx <= len(normalized):
                result_value = normalized[idx - 1][0]
                if confirmation is not None:
                    _print_confirmation(confirmation, result_value, indent)
                return result_value
        except ValueError:
            pass

        # Check if it matches a value or label
        for value, label in normalized:
            if value.lower() == response.lower() or label.lower() == response.lower():
                if confirmation is not None:
                    _print_confirmation(confirmation, value, indent)
                return value

        print(f"{prefix}Invalid selection. Please enter 1-{len(normalized)}.")


def prompt_choice(
    message: str,
    options: Sequence[tuple[str, str]],
    *,
    default: str | None = None,
    indent: int = 0,
    confirmation: str | Callable[[str], str] | None = None,
) -> str:
    """Prompt user to select from options using arrow keys.

    Displays a list of options that can be navigated with up/down arrows.
    Press Enter to confirm selection.

    Args:
        message: The prompt message to display.
        options: Sequence of (value, label) tuples. Value is returned, label is displayed.
        default: Default value to pre-select.
        indent: Number of spaces to indent.
        confirmation: Optional confirmation message. Can be a static string or
            a callable that receives the result and returns a message.

    Returns:
        The selected value (first element of tuple).

    Raises:
        UserCancelled: If the user cancels (Ctrl+C/EOF).
    """
    options_list = list(options)
    selected_index = 0

    # Find default index
    if default:
        for i, (value, _) in enumerate(options_list):
            if value == default:
                selected_index = i
                break

    prefix = " " * indent

    def get_formatted_text() -> FormattedText:
        """Generate the formatted text for the menu."""
        result: list[tuple[str, str]] = []
        # Message line
        result.append(("class:message", f"{prefix}{message}\n"))
        # Options
        for i, (value, label) in enumerate(options_list):
            if i == selected_index:
                result.append(("class:selected", f"{prefix}● {i + 1}. {label}\n"))
            else:
                result.append(("", f"{prefix}  {i + 1}. {label}\n"))
        return FormattedText(result)

    # Key bindings
    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def move_up(event: KeyPressEvent) -> None:
        nonlocal selected_index
        selected_index = (selected_index - 1) % len(options_list)

    @kb.add("down")
    @kb.add("j")
    def move_down(event: KeyPressEvent) -> None:
        nonlocal selected_index
        selected_index = (selected_index + 1) % len(options_list)

    @kb.add("enter")
    def accept(event: KeyPressEvent) -> None:
        event.app.exit(result=options_list[selected_index][0])

    @kb.add("c-c")
    @kb.add("c-d")
    def cancel(event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    # Number keys for direct selection
    for num in range(1, min(10, len(options_list) + 1)):

        @kb.add(str(num))
        def select_number(event: KeyPressEvent, n: int = num) -> None:
            if n <= len(options_list):
                event.app.exit(result=options_list[n - 1][0])

    # Style
    style = Style.from_dict(
        {
            "message": "ansicyan bold",
            "selected": "noreverse",
        }
    )

    # Create application
    app: Application[str | None] = Application(
        layout=Layout(Window(FormattedTextControl(get_formatted_text, show_cursor=False))),
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=False,
    )

    try:
        result = app.run()
    except (KeyboardInterrupt, EOFError):
        raise UserCancelled from None

    if result is None:
        raise UserCancelled

    if confirmation is not None:
        _print_confirmation(confirmation, result, indent)
    return result


@overload
def prompt_multiselect(
    message: str,
    options: Sequence[tuple[str, str]],
    *,
    defaults: Sequence[str] | None = ...,
    min_choices: int = ...,
    indent: int = ...,
    confirmation: str | Callable[[list[str]], str] | None = ...,
    sub_labels: Mapping[str, Sequence[str]] | None = ...,
    allow_skip: Literal[False] = ...,
) -> list[str]: ...


@overload
def prompt_multiselect(
    message: str,
    options: Sequence[tuple[str, str]],
    *,
    defaults: Sequence[str] | None = ...,
    min_choices: int = ...,
    indent: int = ...,
    confirmation: str | Callable[[list[str]], str] | None = ...,
    sub_labels: Mapping[str, Sequence[str]] | None = ...,
    allow_skip: Literal[True],
) -> list[str] | None: ...


def prompt_multiselect(
    message: str,
    options: Sequence[tuple[str, str]],
    *,
    defaults: Sequence[str] | None = None,
    min_choices: int = 1,
    indent: int = 0,
    confirmation: str | Callable[[list[str]], str] | None = None,
    sub_labels: Mapping[str, Sequence[str]] | None = None,
    allow_skip: bool = False,
) -> list[str] | None:
    """Prompt user to toggle multiple options with checkboxes.

    Displays a checkbox list navigable with up/down arrows. Space toggles the
    current row, 'a' toggles all rows, Enter confirms the selection.

    Args:
        message: The prompt message to display.
        options: Sequence of (value, label) tuples. Value is returned, label is displayed.
        defaults: Optional iterable of values to start toggled. Unknown values are ignored.
        min_choices: Minimum number of toggled entries required to confirm.
        indent: Number of spaces to indent.
        confirmation: Optional confirmation message. Can be a static string or a callable
            that receives the selected values and returns a message.
        sub_labels: Optional map from option value to a list of read-only sub-labels
            rendered indented beneath the option. Useful for showing nested context
            (e.g. items contained in a group) without making them individually toggleable.
        allow_skip: If True, binds ``s`` to exit the prompt with ``None``, signaling
            the caller to skip whatever this prompt was driving. Adds ``[s] skip`` to
            the hint footer.

    Returns:
        Selected values in option order. Returns ``None`` only when ``allow_skip`` is
        True and the user pressed ``s``.

    Raises:
        UserCancelled: If the user cancels (Ctrl+C/EOF).
    """
    options_list = list(options)
    if not options_list:
        return []

    default_set = set(defaults or ())
    toggled: set[int] = {i for i, (value, _) in enumerate(options_list) if value in default_set}
    selected_index = 0
    show_min_error = False

    prefix = " " * indent
    hint_parts = ["[space] toggle", "[a] all/none", "[↑↓/jk] move", "[enter] confirm"]
    if allow_skip:
        hint_parts.append("[s] skip")
    hint_parts.append("[^C] cancel")
    hint = "  ".join(hint_parts)

    # Width of "● [x] N. " so sub-label lines align under the option label.
    max_num_width = len(str(len(options_list)))
    sub_label_pad = " " * (2 + 4 + max_num_width + 2)  # cursor+space, "[x] ", "N.", trailing space

    def get_formatted_text() -> FormattedText:
        result: list[tuple[str, str]] = []
        result.append(("class:message", f"{prefix}{message}\n"))
        for i, (value, label) in enumerate(options_list):
            cursor = "●" if i == selected_index else " "
            checkbox = "[x]" if i in toggled else "[ ]"
            number = f"{i + 1}.".rjust(max_num_width + 1)
            line = f"{prefix}{cursor} {checkbox} {number} {label}\n"
            style_class = "class:selected" if i == selected_index else ""
            result.append((style_class, line))
            if sub_labels:
                for sub in sub_labels.get(value, ()):
                    result.append(("class:sublabel", f"{prefix}{sub_label_pad}- {sub}\n"))
        if show_min_error:
            result.append(("class:error", f"{prefix}Select at least {min_choices}.\n"))
        result.append(("class:hint", f"{prefix}{hint}\n"))
        return FormattedText(result)

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def move_up(event) -> None:  # type: ignore[no-untyped-def]
        nonlocal selected_index, show_min_error
        selected_index = (selected_index - 1) % len(options_list)
        show_min_error = False

    @kb.add("down")
    @kb.add("j")
    def move_down(event) -> None:  # type: ignore[no-untyped-def]
        nonlocal selected_index, show_min_error
        selected_index = (selected_index + 1) % len(options_list)
        show_min_error = False

    @kb.add("space")
    def toggle_current(event) -> None:  # type: ignore[no-untyped-def]
        nonlocal show_min_error
        if selected_index in toggled:
            toggled.remove(selected_index)
        else:
            toggled.add(selected_index)
        show_min_error = False

    @kb.add("a")
    def toggle_all(event) -> None:  # type: ignore[no-untyped-def]
        nonlocal show_min_error
        if len(toggled) == len(options_list):
            toggled.clear()
        else:
            toggled.update(range(len(options_list)))
        show_min_error = False

    @kb.add("enter")
    def accept(event) -> None:  # type: ignore[no-untyped-def]
        nonlocal show_min_error
        if len(toggled) < min_choices:
            show_min_error = True
            return
        values = [options_list[i][0] for i in sorted(toggled)]
        event.app.exit(result=values)

    was_skipped = False

    if allow_skip:

        @kb.add("s")
        def skip(event) -> None:  # type: ignore[no-untyped-def]
            nonlocal was_skipped
            was_skipped = True
            event.app.exit(result=None)

    @kb.add("c-c")
    @kb.add("c-d")
    def cancel(event) -> None:  # type: ignore[no-untyped-def]
        event.app.exit(result=None)

    style = Style.from_dict(
        {
            "message": "ansicyan bold",
            "selected": "noreverse",
            "hint": "ansiblue italic",
            "error": "ansired",
            "sublabel": "italic",
        }
    )

    app: Application[list[str] | None] = Application(
        layout=Layout(Window(FormattedTextControl(get_formatted_text, show_cursor=False))),
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=False,
    )

    try:
        result = app.run()
    except (KeyboardInterrupt, EOFError):
        raise UserCancelled from None

    if was_skipped:
        return None
    if result is None:
        raise UserCancelled

    if confirmation is not None:
        _print_confirmation(confirmation, result, indent)
    return result
