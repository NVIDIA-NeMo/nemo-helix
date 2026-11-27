# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Choose which modules train for ``partial_sft``, with the same rules as Automodel's selectors.

Unsloth has no option for this, so Helix sets ``requires_grad`` itself after the model loads and
before the trainer starts. Matching Automodel's rules keeps one job spec training the same
modules on both backends:

- A selector picks one module by exact ``path``, or every module whose path matches ``glob``
  (``fnmatch``, so ``*`` also crosses ``.``), together with everything under it.
- ``freeze_modules`` apply first, then ``unfreeze_modules``, which win on overlap.
- A selector that matches no parameters fails the job.

Torch-free: it only calls ``named_modules``, ``parameters``, and ``requires_grad_``.
"""

import fnmatch
from collections.abc import Sequence
from typing import Any

from nhx.unsloth.schemas import ModuleSelector

# PEFT nests the base model under this prefix; selectors name modules as the base model does.
_PEFT_PREFIX = "base_model.model."
# PEFT keeps the trainable copy of a modules_to_save module under this path segment.
MODULES_TO_SAVE_SEGMENT = ".modules_to_save."


def module_path(name: str) -> str:
    """A module's path as selectors name it: without PEFT's prefix or ``torch.compile``'s ``_orig_mod.``."""
    return name.removeprefix(_PEFT_PREFIX).replace("_orig_mod.", "")


def _matches(selector: ModuleSelector, path: str) -> bool:
    if selector.path is not None:
        return path == selector.path
    return fnmatch.fnmatchcase(path, selector.glob or "")


def _selected_modules(model: Any, selectors: Sequence[ModuleSelector], field_name: str) -> list[tuple[str, Any]]:
    """Every module each selector picks, as ``(path, module)``. Fails on a selector that picks no parameters."""
    named = [(module_path(name), module) for name, module in model.named_modules() if name]
    selected: list[tuple[str, Any]] = []
    for selector in selectors:
        hits = [(path, module) for path, module in named if _matches(selector, path)]
        if not any(True for _, module in hits for _ in module.parameters()):
            raise ValueError(
                f"training.{field_name} selector {selector.model_dump(exclude_none=True)} matched no parameters. "
                "Selectors match the full module path, e.g. 'model.layers.0.mlp'."
            )
        selected.extend(hits)
    return selected


def unfrozen_module_paths(model: Any, selectors: Sequence[ModuleSelector]) -> list[str]:
    """The outermost modules the unfreeze selectors pick, as PEFT ``modules_to_save`` entries.

    A LoRA adapter keeps fully trained modules only as ``modules_to_save``, so with LoRA the
    unfrozen modules go there instead of only having ``requires_grad`` set. A module inside
    another picked module is already covered by its parent.
    """
    paths = sorted({path for path, _ in _selected_modules(model, selectors, "unfreeze_modules")})
    return [path for path in paths if not any(path.startswith(f"{parent}.") for parent in paths)]


def set_trainable_modules(
    model: Any,
    freeze: Sequence[ModuleSelector] | None,
    unfreeze: Sequence[ModuleSelector] | None,
) -> None:
    """Freeze, then unfreeze, each selected module with everything under it, and fail if nothing trains.

    PEFT ``modules_to_save`` copies stay trainable, because they hold the LoRA run's unfrozen modules.
    """
    for _, module in _selected_modules(model, freeze or (), "freeze_modules"):
        module.requires_grad_(False)
    for _, module in _selected_modules(model, unfreeze or (), "unfreeze_modules"):
        module.requires_grad_(True)
    for name, module in model.named_modules():
        if MODULES_TO_SAVE_SEGMENT in f"{name}.":
            module.requires_grad_(True)
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("training.freeze_modules leaves no trainable parameters.")
