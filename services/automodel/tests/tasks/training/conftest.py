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
import json
import sys
from pathlib import Path
from types import SimpleNamespace
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


def _automodel_cross_encoder_class(config: Any) -> type | None:
    """Automodel v0.7.0's cross-encoder dispatch: ``SUPPORTED_BACKBONES`` plus ``supports_config``.

    Only the dispatch is stood in for. The config it gets is the one ``transformers`` loaded.
    """
    model_type = str(config.model_type).lower()
    if model_type in ("llama", "llama_bidirec"):
        return LlamaBidirectionalForSequenceClassification
    if model_type == "qwen3" and {"Qwen3ForCausalLM", "Qwen3RerankerForCausalReranking"} & set(
        config.architectures or ()
    ):
        return Qwen3RerankerForCausalReranking
    return None


def _load_checkpoint_config(model_path: Path, *, trust_remote_code: bool) -> SimpleNamespace:
    """``config.json`` as attributes, for runs without the opt-in ``automodel-tests`` group.

    Like ``AutoConfig.from_pretrained``, a missing ``config.json`` raises ``OSError`` and bad JSON ``ValueError``.
    """
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    return SimpleNamespace(model_type=config.get("model_type", ""), architectures=config.get("architectures"))


@pytest.fixture(autouse=True)
def automodel_cross_encoder_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``nemo_automodel``, answer the registry's dispatch question the way Automodel would.

    Without ``transformers``, read the checkpoint config from its JSON as well.
    """
    registry = "nhx.automodel.tasks.training.backends.registry"
    if importlib.util.find_spec("nemo_automodel") is None:
        monkeypatch.setattr(f"{registry}._automodel_cross_encoder_class", _automodel_cross_encoder_class)
    if importlib.util.find_spec("transformers") is None:
        monkeypatch.setattr(f"{registry}._load_checkpoint_config", _load_checkpoint_config)
