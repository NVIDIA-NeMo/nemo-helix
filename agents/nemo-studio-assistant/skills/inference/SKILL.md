---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: inference
description: "Explain inference providers and models in NeMo Helix and manage providers with nemo_api(resource='inference.providers'). Use when the user asks how to connect a model, which models are available, or wants to add, inspect, or remove a provider."
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---
# Inference providers and models

An inference provider connects the workspace to a model endpoint, for example
build.nvidia.com, an OpenAI-compatible API, or a self-hosted NIM. Models from
a provider then show up for chat, agents, evaluation, and guardrails. Most
other features need at least one working model, so this is usually the first
thing to set up.

## Explaining

- Check what exists first: `nemo_api(resource="inference.providers", action="list", ...)`
  and `nemo_api(resource="models", action="list", ...)`.
- If there are no providers, say so and link to `inference_providers`.
- If models exist, suggest trying one in `model_chat`.
- A VirtualModel is a stable name that routes to one or more models and can
  apply guardrails. Link to `virtual_models` when the user asks about routing
  or safety on a model endpoint.

## Acting on request

- A provider usually needs an API key. Store it as a secret first (see the
  `secrets` skill) and reference it from the provider.
- Use `nemo_api` with resource `inference.providers` for `list`, `retrieve`,
  `create`, `update`, and `delete`, passing
  `workspace="<active request workspace>"` on every call.
- Use exactly the host URL and description the user gave. After creating or
  updating, retrieve the provider and confirm those fields before reporting
  success.
