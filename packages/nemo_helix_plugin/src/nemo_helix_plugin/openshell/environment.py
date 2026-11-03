# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenShell-only settings stored in an environment's provider-specific settings."""

from typing import Annotated

from nemo_helix_plugin.refs import ENTITY_REF_PATTERN
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

CredentialSlot = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
ProviderBindingName = Annotated[str, StringConstraints(min_length=1, pattern=r"^\S+$")]


class ProviderFulfillment(BaseModel):
    """Select an injection profile and map its credential slots to Helix secret references.

    The OpenShell backend checks profile approval and creator access.
    """

    model_config = ConfigDict(extra="forbid")

    profile: ProviderBindingName = Field(description="Operator-approved credential injection profile identifier.")
    credentials: dict[CredentialSlot, Annotated[str, StringConstraints(pattern=ENTITY_REF_PATTERN)]] = Field(
        min_length=1,
        description="Profile credential slot -> Helix secret reference (name or workspace/name), not a secret value.",
    )


class OpenShellEnvironmentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: dict[ProviderBindingName, ProviderFulfillment] = Field(default_factory=dict)

    def validate_slots(self) -> None:
        slots: set[str] = set()
        for provider in self.providers.values():
            duplicates = slots.intersection(provider.credentials)
            if duplicates:
                raise ValueError(f"Providers declare duplicate credential slots: {', '.join(sorted(duplicates))}.")
            slots.update(provider.credentials)
