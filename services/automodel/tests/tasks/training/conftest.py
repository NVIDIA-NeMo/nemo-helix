# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import the training backend modules once, under a ``nemo_automodel`` stub that is then removed.

``config.py`` imports ``nemo_automodel`` at module level, and that package exists only in the
training image. The stub is in ``sys.modules`` just long enough for that import. Left behind, it
would make every later test on the xdist worker see a fake ``nemo_automodel``; for example,
``pytest.importorskip("nemo_automodel")`` would stop skipping. The imported modules stay cached
and keep their references to the stub.
"""

import importlib
import importlib.util
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

_STUBBED = ("nemo_automodel", "nemo_automodel._transformers", "nemo_automodel._transformers.registry")
_IMPORTED_UNDER_STUB = (
    "nhx.automodel.tasks.training.backends.config",
    "nhx.automodel.tasks.training.backends.backend",
)

if importlib.util.find_spec("nemo_automodel") is None:
    stubbed = [name for name in _STUBBED if name not in sys.modules]
    for name in stubbed:
        sys.modules[name] = MagicMock()
    try:
        for module in _IMPORTED_UNDER_STUB:
            importlib.import_module(module)
    finally:
        for name in stubbed:
            del sys.modules[name]


class Qwen3RerankerForCausalReranking:
    """Stands in for Automodel's yes/no reranker class, a causal LM."""

    @classmethod
    def can_generate(cls) -> bool:
        return True


class LlamaBidirectionalForSequenceClassification:
    """Stands in for Automodel's score-head reranker class."""

    @classmethod
    def can_generate(cls) -> bool:
        return False


def _automodel_cross_encoder_class(config: dict[str, Any]) -> type | None:
    """Automodel v0.7.0's cross-encoder dispatch: ``SUPPORTED_BACKBONES`` plus ``supports_config``."""
    model_type = str(config.get("model_type", "")).lower()
    if model_type in ("llama", "llama_bidirec"):
        return LlamaBidirectionalForSequenceClassification
    if model_type == "qwen3" and {"Qwen3ForCausalLM", "Qwen3RerankerForCausalReranking"} & set(
        config.get("architectures") or ()
    ):
        return Qwen3RerankerForCausalReranking
    return None


def _automodel_bi_encoder_pooling(_model_path: object) -> tuple[str, bool]:
    """Automodel's bi-encoder pooling for both supported bases, whose files say mean then normalize."""
    return "avg", True


@pytest.fixture(autouse=True)
def automodel_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``nemo_automodel``, answer the questions Helix asks Automodel the way it would."""
    if importlib.util.find_spec("nemo_automodel") is None:
        monkeypatch.setattr(
            "nhx.automodel.tasks.training.backends.registry._automodel_cross_encoder_class",
            _automodel_cross_encoder_class,
        )
        monkeypatch.setattr(
            "nhx.automodel.tasks.training.backends.checkpoints._automodel_bi_encoder_pooling",
            _automodel_bi_encoder_pooling,
        )
