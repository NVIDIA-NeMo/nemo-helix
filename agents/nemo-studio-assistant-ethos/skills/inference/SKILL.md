---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: inference
description: NeMo Helix inference provider registration lifecycle through the platform SDK (secret -> create temp provider -> verify -> delete -> create final provider).
---
# Inference provider tasks

- For provider registration tasks, use this sequence:
  1) Create API key secret
  2) Create temporary provider
  3) List providers
  4) Get provider details
  5) Delete temporary provider
  6) Create final verification provider
- Use `nemo_api` with resource `secrets` for the API key secret, passing
  `workspace="<active request workspace>"`.
- Use `nemo_api` with resource `inference.providers` for provider CRUD, passing
  `workspace="<active request workspace>"` on every call.
- Ensure final provider uses exactly the requested host URL and description.
- Retrieve or list the final provider and compare its host URL and description
  with the requested values before reporting success.
