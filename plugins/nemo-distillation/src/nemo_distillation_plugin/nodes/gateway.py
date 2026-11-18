# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gateway model name for a deployment this workflow created.

A LoRA deployment serves the base model. Its gateway name is
``{workspace}/{base}&adapters/{workspace}/{adapter}``, and both parts are names
this workflow assigned (the adapter) or the workflow's student model (the base).
A full-weights deployment's gateway name is ``{workspace}/{output}``. The base
deployment's gateway name is the student model itself.
"""

from __future__ import annotations

from typing import Any


def qualify(workspace: str, model: str) -> str:
    return model if "/" in model else f"{workspace}/{model}"


def gateway_model(workspace: str, inputs: dict[str, Any]) -> str:
    qualified = qualify(workspace, str(inputs["model"]))
    if inputs.get("role") == "base" or not inputs.get("lora_enabled"):
        return qualified
    namespace, name = qualified.split("/", 1)
    return f"{namespace}/{name}&adapters/{workspace}/{inputs['adapter']}"
