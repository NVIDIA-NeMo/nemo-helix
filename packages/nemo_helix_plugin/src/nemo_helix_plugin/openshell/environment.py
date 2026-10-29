# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenShell-only settings stored in an environment's provider-specific settings."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

CredentialSlot = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
ProviderBindingName = Annotated[str, StringConstraints(min_length=1, pattern=r"^\S+$")]


class ProviderFulfillment(BaseModel):
    """Select an injection profile and map its credential slots to environment secret aliases.

    Agent environment resolution replaces the aliases with secret references for
    the OpenShell backend. The backend checks profile approval and creator access.
    """

    model_config = ConfigDict(extra="forbid")

    profile: ProviderBindingName = Field(description="Operator-approved credential injection profile identifier.")
    credentials: dict[CredentialSlot, CredentialSlot] = Field(
        min_length=1,
        description="Profile credential slot -> key in this EnvironmentSpec's secrets map, not a secret value.",
    )


class OpenShellEnvironmentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: dict[ProviderBindingName, ProviderFulfillment] = Field(default_factory=dict)

    def validate_secrets(self, secrets: dict[str, str]) -> None:
        slots: set[str] = set()
        for name, provider in self.providers.items():
            missing = sorted(set(provider.credentials.values()) - secrets.keys())
            if missing:
                raise ValueError(f"Provider {name!r} references unknown secret entries: {', '.join(missing)}.")
            duplicates = slots.intersection(provider.credentials)
            if duplicates:
                raise ValueError(f"Providers declare duplicate credential slots: {', '.join(sorted(duplicates))}.")
            slots.update(provider.credentials)
            for alias in provider.credentials.values():
                if not secrets[alias].strip():
                    raise ValueError(f"Provider {name!r} references empty secret entry {alias!r}.")
