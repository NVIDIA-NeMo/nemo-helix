# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for prompt_multiselect.

Tests drive the underlying ``prompt_toolkit.Application`` with a pipe input
that pushes raw key sequences (e.g. ``\\x1b[B`` for the down arrow, ``\\r``
for enter, ``\\x03`` for ctrl-c) and a ``DummyOutput`` that swallows rendering.
"""

from __future__ import annotations

import pytest
from nemo_helix_ext.ui.prompts import UserCancelled, prompt_multiselect
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

ENTER = "\r"
SPACE = " "
DOWN = "\x1b[B"
UP = "\x1b[A"
CTRL_C = "\x03"
CTRL_D = "\x04"


def _run(keys: str, **kwargs) -> list[str] | None:
    """Drive prompt_multiselect with a pre-recorded key sequence."""
    options = kwargs.pop("options", [("a", "Alpha"), ("b", "Beta"), ("c", "Gamma")])
    with create_pipe_input() as pipe_input:
        pipe_input.send_text(keys)
        with create_app_session(input=pipe_input, output=DummyOutput()):
            return prompt_multiselect(
                message="Pick:",
                options=options,
                **kwargs,
            )


def test_toggle_single_then_confirm():
    # space toggles current row (Alpha), enter confirms
    result = _run(SPACE + ENTER)
    assert result == ["a"]


def test_toggle_multiple_via_arrows():
    # toggle Alpha, down, down, toggle Gamma, enter
    result = _run(SPACE + DOWN + DOWN + SPACE + ENTER)
    assert result == ["a", "c"]


def test_toggle_all_with_a():
    # 'a' toggles all on, enter
    result = _run("a" + ENTER)
    assert result == ["a", "b", "c"]


def test_toggle_all_then_off_blocks_enter():
    # 'a' toggles all on, 'a' again toggles all off. The first ENTER should be
    # blocked because min_choices=1; SPACE+ENTER then succeeds with a single
    # selection (verifies the block actually fires).
    result = _run("a" + "a" + ENTER + SPACE + ENTER)
    assert result == ["a"]


def test_defaults_preselect():
    result = _run(ENTER, defaults=["b"])
    assert result == ["b"]


def test_defaults_can_be_toggled_off():
    # b is preselected; toggle a on, move to b and toggle off, enter
    result = _run(SPACE + DOWN + SPACE + ENTER, defaults=["b"])
    assert result == ["a"]


def test_unknown_defaults_ignored():
    # 'zzz' isn't a real option, should be silently dropped
    result = _run(SPACE + ENTER, defaults=["zzz"])
    assert result == ["a"]


def test_min_choices_blocks_empty_confirm():
    # No toggle, enter → still no selection, so we toggle one and confirm
    result = _run(ENTER + SPACE + ENTER, min_choices=1)
    assert result == ["a"]


def test_ctrl_c_cancels():
    with pytest.raises(UserCancelled):
        _run(CTRL_C)


def test_ctrl_d_cancels():
    with pytest.raises(UserCancelled):
        _run(CTRL_D)


def test_empty_options_returns_empty_list():
    # Edge case: no options → returns [] without prompting
    result = prompt_multiselect(message="Pick:", options=[])
    assert result == []


def test_returns_values_in_option_order():
    # Toggle in reverse order; result should still be in option order
    # down, down, toggle Gamma, up, up, toggle Alpha, enter
    result = _run(DOWN + DOWN + SPACE + UP + UP + SPACE + ENTER)
    assert result == ["a", "c"]


def test_allow_skip_returns_none_when_s_pressed():
    # 's' should short-circuit even when toggles are set; caller sees None.
    with create_pipe_input() as pipe_input:
        pipe_input.send_text(SPACE + "s")
        with create_app_session(input=pipe_input, output=DummyOutput()):
            result = prompt_multiselect(
                message="Pick:",
                options=[("a", "Alpha"), ("b", "Beta")],
                allow_skip=True,
            )
    assert result is None


def test_s_is_ignored_when_skip_not_allowed():
    # Without allow_skip, 's' is an inert keypress; enter still confirms normally.
    result = _run(SPACE + "s" + ENTER)
    assert result == ["a"]


def test_sub_labels_do_not_change_returned_values():
    # Sub-labels are display-only — they must not alter the values produced
    # by the underlying option toggles.
    result = _run(
        SPACE + ENTER,
        sub_labels={"a": ["child-1", "child-2"], "b": ["only"]},
    )
    assert result == ["a"]
