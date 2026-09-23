# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# provider is the single category and states what the secret is for: the model
# providers carry an API key; 'nhx' is NHX Intake (workspace token as `yaml`);
# 'openshift' is a user's OpenShift/Kubernetes bearer token (as `key`) that
# dispatch turns into a per-evaluation kubeconfig so the run acts as that user
# (lets a user run custom-image / direct-mode Harbor tasks with their own
# cluster rights — no service-account NetworkPolicy grant). See
# scaled_evals.dispatch.sandbox_k8s. 'switchyard' is an evaluation-scoped
# client token for an operator-approved external Switchyard endpoint.
CredentialProvider = Literal["openai", "anthropic", "nvidia", "nhx", "openshift", "switchyard"]
# Which write-once payload was supplied. Both are secret material (the whole
# resource is a secrets store); `key` is a single-string secret (model API
# key), `yaml` is a structured secret blob (intake workspace token).
PayloadKind = Literal["key", "yaml"]


def _require_one_payload(key: str | None, yaml: str | None) -> PayloadKind:
    """Enforce the write-once "key XOR yaml" rule and report which was
    supplied. Raises ValueError (→ 422) when neither or both are set."""
    if bool(key) == bool(yaml):
        raise ValueError("provide exactly one of 'key' or 'yaml'")
    return "key" if key else "yaml"


# Request body: POST /v1/credentials
class CredentialCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    provider: CredentialProvider
    key: str | None = Field(default=None, min_length=1)
    yaml: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "CredentialCreate":
        _require_one_payload(self.key, self.yaml)
        return self

    @property
    def payload(self) -> str:
        return self.key or self.yaml  # type: ignore[return-value]

    @property
    def payload_kind(self) -> PayloadKind:
        return _require_one_payload(self.key, self.yaml)


# Request body: PATCH /v1/credentials/{id} (rename only)
class CredentialRename(BaseModel):
    name: str = Field(min_length=1, max_length=200)


# Request body: POST /v1/credentials/{id}/rotate (replace encrypted payload)
class CredentialRotate(BaseModel):
    key: str | None = Field(default=None, min_length=1)
    yaml: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "CredentialRotate":
        _require_one_payload(self.key, self.yaml)
        return self

    @property
    def payload(self) -> str:
        return self.key or self.yaml  # type: ignore[return-value]

    @property
    def payload_kind(self) -> PayloadKind:
        return _require_one_payload(self.key, self.yaml)


# Response: GET /v1/credentials/{id} and list items — metadata only, never
# the plaintext payload.
class Credential(BaseModel):
    id: str
    name: str
    provider: CredentialProvider
    payload_kind: PayloadKind
    fingerprint: str
    created_at: datetime
    updated_at: datetime


# Response body: POST /v1/credentials
class CredentialCreateResponse(Credential):
    links: dict[str, str]


class CredentialVerifyResponse(BaseModel):
    id: str
    verified: bool | None
    reason: str
