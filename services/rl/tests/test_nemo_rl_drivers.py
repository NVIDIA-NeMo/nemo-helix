# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-level checks that the drivers tear the progress logger down.

These are tripwires, not behaviour tests. The drivers cannot be imported outside
the training image -- they pull in nemo_rl and omegaconf at module scope -- so
the wiring is asserted against the AST instead.

It is worth asserting at all because the failure is silent: NeMo-RL never closes
the loggers it is handed, so if these calls are dropped the final training step
stops being reported and every unit test still passes.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

DRIVERS = Path(__file__).resolve().parents[1] / "src/nhx/rl/tasks/training/backends/nemo_rl"


def _closes_logger_in_finally(source: str) -> bool:
    """Whether some `try/finally` closes `customizer_logger` in its finally body."""
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Try):
            continue
        for stmt in node.finalbody:
            for inner in ast.walk(stmt):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "close"
                    and isinstance(inner.func.value, ast.Name)
                    and inner.func.value.id == "customizer_logger"
                ):
                    return True
    return False


@pytest.mark.parametrize(
    "source,expected",
    [
        ("try:\n    train()\nfinally:\n    customizer_logger.close()\n", True),
        # Guarded call — how the drivers actually write it.
        ("try:\n    train()\nfinally:\n    if customizer_logger:\n        customizer_logger.close()\n", True),
        # Present, but not on the abnormal-exit path.
        ("try:\n    train()\nfinally:\n    pass\ncustomizer_logger.close()\n", False),
        ("try:\n    train()\nfinally:\n    other_logger.close()\n", False),
        ("try:\n    train()\nfinally:\n    customizer_logger.flush()\n", False),
    ],
)
def test_detector_discriminates(source: str, expected: bool) -> None:
    """The tripwire is only worth having if it can actually trip."""
    assert _closes_logger_in_finally(source) is expected


@pytest.mark.parametrize("driver", ["dpo_driver.py"])
def test_driver_closes_the_progress_logger_in_a_finally(driver: str) -> None:
    """`finally`, not the happy path: an aborted run is when the flush matters."""
    source = (DRIVERS / driver).read_text()

    assert _closes_logger_in_finally(source), (
        f"{driver} must close customizer_logger from a finally block; "
        "NeMo-RL does not close loggers, and __del__ does not run on abnormal exit"
    )


def _reads_via_defaulted_getattr(source: str, field: str) -> bool:
    """Whether `field` is read with a three-argument getattr and never as an attribute."""
    via_getattr = False
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr == field:
            return False
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) == 3
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == field
        ):
            via_getattr = True
    return via_getattr


@pytest.mark.parametrize(
    "source,expected",
    [
        ("x = getattr(config.dpo, 'steps_per_epoch', None)\n", True),
        # Attribute access on an undeclared extra: AttributeError at startup.
        ("x = config.dpo.steps_per_epoch\n", False),
        # No default, so it raises exactly as attribute access would.
        ("x = getattr(config.dpo, 'steps_per_epoch')\n", False),
        # Never read at all.
        ("x = 1\n", False),
    ],
)
def test_optional_field_detector_discriminates(source: str, expected: bool) -> None:
    assert _reads_via_defaulted_getattr(source, "steps_per_epoch") is expected


@pytest.mark.parametrize("driver", ["dpo_driver.py"])
def test_driver_reads_steps_per_epoch_defensively(driver: str) -> None:
    """It is an undeclared extra, so a config compiled elsewhere simply omits it.

    pydantic raises AttributeError for a missing extra, and this read happens at
    driver startup -- before `for_schedule` can apply the fallback that derives
    steps_per_epoch from max_steps and num_epochs. Hard attribute access turns
    that fallback into dead code and the missing key into a crash.
    """
    source = (DRIVERS / driver).read_text()

    assert _reads_via_defaulted_getattr(source, "steps_per_epoch"), (
        f"{driver} must read steps_per_epoch via getattr with a None default; "
        "it is an extra=allow field that a config need not carry"
    )


def _for_schedule_keywords(source: str) -> set[str]:
    """Keyword names passed to any `NemoRLLogger.for_schedule(...)` call."""
    return {
        keyword.arg
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "for_schedule"
        for keyword in node.keywords
        if keyword.arg is not None
    }


def test_for_schedule_keyword_detector_discriminates() -> None:
    """The tripwire is only worth having if it can actually trip."""
    assert _for_schedule_keywords("x = NemoRLLogger.for_schedule(run_facts=f, max_steps=1)\n") == {
        "run_facts",
        "max_steps",
    }
    assert _for_schedule_keywords("x = NemoRLLogger.other(run_facts=f)\n") == set()


@pytest.mark.parametrize("driver", ["grpo_driver.py", "dpo_driver.py"])
def test_driver_states_which_algorithm_it_is(driver: str) -> None:
    """`backend` is `nemo_rl` for both, so run_facts is the only thing telling them apart.

    Per-algorithm, and only the driver has the compiled config to read it from.
    """
    assert "run_facts" in _for_schedule_keywords((DRIVERS / driver).read_text())
