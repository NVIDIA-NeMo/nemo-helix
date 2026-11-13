# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""partial_sft module selection on a torch-free stand-in for a model's module tree."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from nhx.unsloth.schemas import ModuleSelector
from nhx.unsloth.tasks.training.backends.module_selection import set_trainable_modules, unfrozen_module_paths


class _Module:
    """A module with one parameter of its own; ``parameters`` includes its children's, as in torch."""

    def __init__(self, *, has_weight: bool = True) -> None:
        self.children: dict[str, _Module] = {}
        self.weight = SimpleNamespace(requires_grad=True) if has_weight else None

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, _Module]]:
        yield prefix, self
        for name, child in self.children.items():
            yield from child.named_modules(f"{prefix}.{name}" if prefix else name)

    def parameters(self) -> Iterator[SimpleNamespace]:
        if self.weight is not None:
            yield self.weight
        for child in self.children.values():
            yield from child.parameters()

    def requires_grad_(self, requires_grad: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = requires_grad


def _model(*paths: str) -> _Module:
    # Like a real model, the root keeps every weight in its submodules.
    root = _Module(has_weight=False)
    for path in paths:
        module = root
        for name in path.split("."):
            module = module.children.setdefault(name, _Module())
    return root


def _trainable(model: _Module, *paths: str) -> list[bool]:
    modules = dict(model.named_modules())
    return [weight.requires_grad for path in paths if (weight := modules[path].weight) is not None]


_LLM = ("model.embed_tokens", "model.layers.0.mlp", "model.layers.1.mlp", "lm_head")
_PEFT = (
    "base_model.model.model.embed_tokens.original_module",
    "base_model.model.model.embed_tokens.modules_to_save.default",
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A",
    "base_model.model.model.layers.1.self_attn.q_proj.lora_A",
)


@pytest.mark.parametrize(
    ("paths", "freeze", "unfreeze", "checked", "expected"),
    [
        (_LLM, [{"glob": "*"}], [{"path": "model.layers.1.mlp"}], _LLM, [False, False, True, False]),
        (_LLM, [{"glob": "*.layers.0.*"}, {"path": "lm_head"}], None, _LLM, [True, False, True, False]),
        (
            _PEFT,
            [{"glob": "*.layers.0.*"}],
            None,
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A",
                "base_model.model.model.layers.1.self_attn.q_proj.lora_A",
            ),
            [False, True],
        ),
        (
            _PEFT,
            [{"glob": "*"}],
            None,
            (
                "base_model.model.model.embed_tokens.modules_to_save.default",
                "base_model.model.model.embed_tokens.original_module",
            ),
            [True, False],
        ),
    ],
    ids=["only-unfrozen-trains", "freeze-some", "peft-paths-match-base-names", "modules-to-save-copy-wins"],
)
def test_set_trainable_modules(
    paths: tuple[str, ...],
    freeze: list[dict[str, str]],
    unfreeze: list[dict[str, str]] | None,
    checked: tuple[str, ...],
    expected: list[bool],
) -> None:
    """Freeze, then unfreeze (which wins), recursively, with PEFT's prefix and copies handled."""
    model = _model(*paths)
    to_selectors = lambda raw: [ModuleSelector.model_validate(s) for s in raw] if raw else None  # noqa: E731

    set_trainable_modules(model, to_selectors(freeze), to_selectors(unfreeze))

    assert _trainable(model, *checked) == expected


@pytest.mark.parametrize(
    ("freeze", "error"),
    [
        ([{"path": "model.layers.7.mlp"}], "matched no parameters"),
        ([{"glob": "*"}], "no trainable parameters"),
    ],
    ids=["no-match", "nothing-trains"],
)
def test_set_trainable_modules_rejects(freeze: list[dict[str, str]], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        set_trainable_modules(_model(*_LLM), [ModuleSelector.model_validate(s) for s in freeze], None)


def test_unfrozen_module_paths_keeps_the_outermost() -> None:
    """A picked module already covers the picked modules inside it."""
    model = _model("model.embed_tokens.inner", "model.layers.0.mlp", "lm_head")

    paths = unfrozen_module_paths(model, [ModuleSelector(glob="*embed_tokens*"), ModuleSelector(path="lm_head")])

    assert paths == ["lm_head", "model.embed_tokens"]
