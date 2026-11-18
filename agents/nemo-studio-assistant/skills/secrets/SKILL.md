---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: secrets
description: "Explain NeMo Helix secrets and manage them by reference with nemo_api(resource='secrets'). Use when the user asks how to store an API key or credential, or wants to list, create, update, or delete a secret."
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---
# Secrets

Secrets hold credentials such as provider API keys. Other resources refer to a
secret by its reference, `<workspace>/<name>`, so the value itself never
appears in configs, traces, or chat.

## Explaining

- When a user is about to paste an API key into chat, stop them. Point to the
  Secrets page with `studio_link` destination `secrets` and explain that they
  should enter the value there.
- Inference providers are the most common consumer: create the secret first,
  then reference it from the provider.

## Acting on request

- Use `nemo_api` with resource `secrets` and actions `list`, `retrieve`,
  `create`, `update`, or `delete`, passing
  `workspace="<active request workspace>"` on every call.
- Pass only references, names, and metadata in `params`. Never print, echo, or
  ask the user to type a secret value in chat; approval inputs and tool
  results can appear in traces.
- If a create or update needs a real credential value, send the user to the
  Secrets page to enter it, then continue with the reference.
