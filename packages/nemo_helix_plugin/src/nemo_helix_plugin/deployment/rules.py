# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rules that decide whether a deployment request is coherent.

These are behaviour, not shape: predicates over a resolved
:class:`~nemo_helix_plugin.models.types.ModelDeploymentConfig` and the
submit-time validator backends call. They live beside :mod:`schemas` rather than
inside it so that module stays what its docstring claims -- plain models whose
only job is to emit one schema into the merged ``/apis/customization`` spec.
"""

from __future__ import annotations

from nemo_helix_plugin.deployment.schemas import DeploymentParams
from nemo_helix_plugin.models.types import ModelDeploymentConfig

LORA_ENABLED_REQUIRED_MESSAGE = (
    "deployment_config.lora_enabled must be true (or omitted) when training a LoRA adapter. "
    "Setting lora_enabled=false would deploy the base model without LoRA support, "
    "making the trained adapter unservable."
)


def is_unbound_deployment_config(config: ModelDeploymentConfig) -> bool:
    """Whether ``config`` names no model entity, making it a reusable template.

    Both links to a model are optional on a ``ModelDeploymentConfig``: the canonical
    ``model_entity_id`` and the older ``model_spec.model_name`` / ``model_namespace``
    pair. A config carrying neither describes only an engine and an executor, so it
    can be pointed at any model -- the model_entity task binds it to the trained
    model at deploy time.

    This matters for full-weight and lora-merged training, which create their own
    model entity: the entity does not exist when the job is submitted, so naming it
    in a deployment config up front is impossible. An unbound config is the only way
    to reuse engine/executor settings for those runs.
    """
    return not config.model_entity_id and not config.model_spec.model_name


def reject_lora_without_lora_enabled(
    deployment_config: str | DeploymentParams | None,
    *,
    trains_lora_adapter: bool,
) -> None:
    """Raise when a LoRA job asks for a deployment that cannot load adapters.

    A LoRA adapter is served by its base model's deployment, so a base deployed with
    ``lora_enabled=false`` would refuse it. Backends call this from a submit-time
    validator, passing their own answer for whether the job trains a standalone
    adapter — that predicate differs per backend, the check does not.

    String references are not checked here: resolving them needs a platform client,
    so the compiler validates those.
    """
    if trains_lora_adapter and isinstance(deployment_config, DeploymentParams) and not deployment_config.lora_enabled:
        raise ValueError(LORA_ENABLED_REQUIRED_MESSAGE)
