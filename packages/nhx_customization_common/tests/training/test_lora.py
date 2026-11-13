# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import types
from pathlib import Path
from typing import Self

import pytest
from nhx.customization_common.training import lora


class _Model:
    path: str
    kwargs: dict[str, object]

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: object) -> Self:
        model = cls()
        model.path = path
        model.kwargs = kwargs
        return model

    def save_pretrained(self, path: Path, safe_serialization: bool = False) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "config.json").write_text(json.dumps({"auto_map": {"AutoConfig": "configuration.Config"}}))

    def merge_and_unload(self) -> Self:
        return self


class _Causal(_Model):
    pass


class _ImageText(_Model):
    pass


class _Tokenizer:
    @staticmethod
    def from_pretrained(path: str, trust_remote_code: bool = False) -> "_Tokenizer":
        del path, trust_remote_code
        return _Tokenizer()

    def save_pretrained(self, path: Path) -> None:
        (Path(path) / "tokenizer_config.json").write_text("{}")
        (Path(path) / "chat_template.jinja").write_text("tokenizer")


class _Processor:
    @staticmethod
    def from_pretrained(path: str, trust_remote_code: bool = False) -> "_Processor":
        del path, trust_remote_code
        return _Processor()

    def save_pretrained(self, path: Path) -> None:
        (Path(path) / "preprocessor_config.json").write_text("{}")
        (Path(path) / "chat_template.jinja").write_text("processor")


class _Peft:
    loaded: _Model | None = None
    adapter: str | None = None

    @staticmethod
    def from_pretrained(model: _Model, adapter: str) -> _Model:
        _Peft.loaded = model
        _Peft.adapter = adapter
        return model


def _module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_ml_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _module("torch", bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "peft", _module("peft", PeftModel=_Peft))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _module(
            "transformers",
            AutoModelForCausalLM=_Causal,
            AutoModelForImageTextToText=_ImageText,
            AutoTokenizer=_Tokenizer,
            AutoProcessor=_Processor,
        ),
    )


@pytest.mark.parametrize(
    ("has_vision_tower", "model_cls"),
    [(False, _Causal), (True, _ImageText)],
)
def test_merge_loads_the_model_class_for_the_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_vision_tower: bool,
    model_cls: type[_Model],
) -> None:
    _Peft.loaded = None
    _Peft.adapter = None
    _install_ml_stubs(monkeypatch)
    monkeypatch.setattr(lora, "_merge_parent", lambda: tmp_path)
    adapter = tmp_path / "adapter"
    output = tmp_path / "output"

    lora.merge_lora_adapter(
        adapter,
        "/models/base",
        output,
        True,
        has_vision_tower=has_vision_tower,
    )

    assert isinstance(_Peft.loaded, model_cls)
    assert _Peft.loaded.path == "/models/base"
    assert _Peft.loaded.kwargs["torch_dtype"] == "bf16"
    assert _Peft.loaded.kwargs["trust_remote_code"] is True
    assert _Peft.adapter == str(adapter)
    assert (output / "config.json").is_file()
    assert (output / "tokenizer_config.json").is_file()
    assert (output / "preprocessor_config.json").is_file() is has_vision_tower
    assert (output / "chat_template.jinja").read_text() == ("processor" if has_vision_tower else "tokenizer")
    assert not list(tmp_path.glob("merged_lora-*"))


@pytest.mark.parametrize(
    ("builtin_model_class", "has_builtin_class", "model_trust_remote_code"),
    [(True, True, False), (True, False, True), (False, True, True)],
    ids=["adapter-uses-builtin-names", "no-builtin-class-for-the-type", "adapter-uses-remote-code-names"],
)
def test_merge_loads_the_class_whose_module_names_the_adapter_uses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    builtin_model_class: bool,
    has_builtin_class: bool,
    model_trust_remote_code: bool,
) -> None:
    """Automodel's Nemotron-H saves adapters as ``model.*``; its remote code names those modules ``backbone.*``."""
    _install_ml_stubs(monkeypatch)
    monkeypatch.setattr(lora, "_merge_parent", lambda: tmp_path)
    monkeypatch.setattr(lora, "_has_builtin_model_class", lambda path: has_builtin_class)
    output = tmp_path / "output"

    lora.merge_lora_adapter(tmp_path / "adapter", "/models/base", output, True, builtin_model_class=builtin_model_class)

    assert _Peft.loaded is not None
    assert _Peft.loaded.kwargs["trust_remote_code"] is model_trust_remote_code
    # Saved from transformers' own class, the checkpoint ships no remote code for auto_map to name.
    assert ("auto_map" in json.loads((output / "config.json").read_text())) is model_trust_remote_code
