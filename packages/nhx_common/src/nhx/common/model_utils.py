# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared model utility helpers used across services."""


# A draft for a better version of this function is in the nhx/services/core/models/src/nhx/core/models/tasks/model_spec/utils.py > is_embedding_model_v2
# Use it instead of this function in services/core/models/src/nhx/core/models/tasks/model_spec/run.py.
# nemo-automodel / nemo-unsloth job compilers use this function as a fallback for embedding model detection.
def is_embedding_model(model_name: str | None) -> bool:
    """Return True when model identifier strongly suggests embedding usage."""
    if model_name is None:
        return False
    return "embed" in model_name.lower()
