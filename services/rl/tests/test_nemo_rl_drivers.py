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

DRIVERS = Path(__file__).resolve().parents[1] / "src/nmp/rl/tasks/training/backends/nemo_rl"


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


def _raises_value_error_after_best_checkpoint_lookup(source: str) -> bool:
    """Whether the `get_best_checkpoint_path() is None` branch raises ValueError."""

    def tests_missing_best_checkpoint(test: ast.expr) -> bool:
        for node in ast.walk(test):
            if not (
                isinstance(node, ast.Compare)
                and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Is)
                and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value is None
                and isinstance(node.left, ast.Call)
                and isinstance(node.left.func, ast.Attribute)
                and node.left.func.attr == "get_best_checkpoint_path"
            ):
                continue
            return True
        return False

    def raises_value_error(stmt: ast.stmt) -> bool:
        """Whether running *stmt* raises ValueError.

        A ``raise`` inside a nested function, lambda, or class body does not run
        when the branch runs, so those scopes are not descended into.
        """
        if (
            isinstance(stmt, ast.Raise)
            and isinstance(stmt.exc, ast.Call)
            and isinstance(stmt.exc.func, ast.Name)
            and stmt.exc.func.id == "ValueError"
        ):
            return True
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return False
        return any(raises_value_error(child) for child in ast.iter_child_nodes(stmt) if isinstance(child, ast.stmt))

    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.If) or not tests_missing_best_checkpoint(node.test):
            continue
        if any(raises_value_error(stmt) for stmt in node.body):
            return True
    return False


def test_postcondition_detector_discriminates() -> None:
    """The tripwire is only worth having if it can actually trip."""
    assert _raises_value_error_after_best_checkpoint_lookup("raise ValueError('x')\n") is False
    assert _raises_value_error_after_best_checkpoint_lookup("checkpointer.get_best_checkpoint_path()\n") is False
    assert (
        _raises_value_error_after_best_checkpoint_lookup(
            "checkpointer.get_best_checkpoint_path()\nraise ValueError('x')\n"
        )
        is False
    )
    assert (
        _raises_value_error_after_best_checkpoint_lookup(
            "if checkpointer.get_best_checkpoint_path() is not None:\n    raise ValueError('x')\n"
        )
        is False
    )
    assert (
        _raises_value_error_after_best_checkpoint_lookup(
            "if checkpointer.get_best_checkpoint_path() is None:\n    raise RuntimeError('x')\n"
        )
        is False
    )
    assert (
        _raises_value_error_after_best_checkpoint_lookup(
            "if checkpointer.get_best_checkpoint_path() is None:\n    raise ValueError('x')\n"
        )
        is True
    )
    # A raise the branch only defines, never executes.
    assert (
        _raises_value_error_after_best_checkpoint_lookup(
            "if checkpointer.get_best_checkpoint_path() is None:\n    def _later():\n        raise ValueError('x')\n"
        )
        is False
    )
    # A raise nested in control flow the branch does execute.
    assert (
        _raises_value_error_after_best_checkpoint_lookup(
            "if checkpointer.get_best_checkpoint_path() is None:\n    if strict:\n        raise ValueError('x')\n"
        )
        is True
    )


def test_grpo_driver_fails_a_run_that_saved_no_checkpoint() -> None:
    """Dynamic sampling can exhaust the dataloader and still let grpo_train return.

    The driver is the last place that knows why, so it has to turn that into a
    non-zero exit. Without the raise the backend sees exit 0, reports training
    success, and the run dies later in publication with a missing checkpoint that
    names no cause.
    """
    source = (DRIVERS / "grpo_driver.py").read_text()

    assert _raises_value_error_after_best_checkpoint_lookup(source), (
        "grpo_driver.py must raise when no checkpoint was saved; NeMo-RL exits 0 after "
        "dynamic sampling finds no trainable group"
    )
    assert "non-zero reward standard deviation" in source
    assert "Training finished without saving a checkpoint." in source


@pytest.mark.parametrize(
    "message,expected_type,expected_detail",
    [
        (
            "Dynamic sampling found no prompt group with non-zero reward standard deviation, "
            "so no training step ran and no checkpoint was saved.",
            "TrainingConfigError",
            "all generations for each prompt earned the same reward",
        ),
        (
            "Training finished without saving a checkpoint.",
            "CheckpointError",
            "Training finished without saving any checkpoint",
        ),
    ],
)
def test_missing_checkpoint_messages_are_classified(message: str, expected_type: str, expected_detail: str) -> None:
    """The driver raises these as ValueError; the rules must map them for the user."""
    from nmp.rl.tasks.training.errors.converter import create_error_details

    details = create_error_details(ValueError(message))
    assert details["type"] == expected_type
    assert expected_detail in details["message"]


@pytest.mark.parametrize("driver", ["grpo_driver.py", "dpo_driver.py"])
def test_driver_states_which_algorithm_it_is(driver: str) -> None:
    """`backend` is `nemo_rl` for both, so run_facts is the only thing telling them apart.

    Per-algorithm, and only the driver has the compiled config to read it from.
    """
    assert "run_facts" in _for_schedule_keywords((DRIVERS / driver).read_text())
