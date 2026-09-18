# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import NoReturn

from fastapi import HTTPException, status


def raise_model_entity_not_found(workspace: str, model_entity_name: str) -> NoReturn:
    """Raise a 404 for a model entity the routing table cannot resolve.

    ``model_entity_name`` may be a LoRA composite
    ``{base}&adapters/{adapter_workspace}/{adapter_name}``. When it is, the message
    names the base model and the adapter separately so the failure is legible
    (e.g. the base isn't served, or the adapter id is wrong).
    """
    if "&adapters/" in model_entity_name:
        base, _, adapter_part = model_entity_name.partition("&adapters/")
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=(
                f"Routing table lookup failed: Model entity not found for base model "
                f"{workspace}/{base} with adapter {adapter_part}"
            ),
        )
    raise HTTPException(
        status.HTTP_404_NOT_FOUND,
        detail=f"Routing table lookup failed: Model entity not found for {workspace}/{model_entity_name}",
    )


def raise_no_providers_for_model_entity(workspace: str, model_entity_name: str) -> NoReturn:
    raise HTTPException(
        status.HTTP_404_NOT_FOUND,
        detail=f"No providers found for model entity {workspace}/{model_entity_name}",
    )


def raise_unresolved_provider_secret(workspace: str, provider_name: str) -> NoReturn:
    raise HTTPException(
        status.HTTP_424_FAILED_DEPENDENCY,
        detail=(f"Could not fetch secret for provider {workspace}/{provider_name}; secret not found or unreachable"),
    )


def raise_virtual_model_not_found(workspace: str, name: str) -> NoReturn:
    """Raise a 404 for a missing :class:`VirtualModel`.

    Inference requests must resolve to a VirtualModel. The platform reconciler
    auto-creates an implicit ``autoprovisioned`` VirtualModel for every served
    model entity, so a missing VM typically means either (a) the underlying
    entity is not currently being served by any provider, or (b) the user is
    referencing a name that doesn't match any entity or operator-defined VM in
    this workspace.
    """
    raise HTTPException(
        status.HTTP_404_NOT_FOUND,
        detail=(
            f"No VirtualModel named '{workspace}/{name}' was found. Inference requests "
            "must resolve to a VirtualModel; one is auto-created for every served "
            "model entity. Check that the model entity exists and is currently "
            "served by a provider, or create a VirtualModel explicitly."
        ),
    )
