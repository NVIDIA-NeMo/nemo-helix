---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: entities
description: "Explain how NeMo Helix models and datasets are registered and manage model entities with nemo_api(resource='models'). Use when the user asks what models or datasets are registered, or wants to register, update, or remove one."
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---
# Models and datasets

Models are registered entities in the workspace, usually created by an
inference provider or a customization job. Datasets are stored as filesets
(see the `files` skill).

## Explaining

- Check what exists with `nemo_api(resource="models", action="list", workspace="<active request workspace>")`
  before suggesting anything.
- Link to `base_models` for the model list and `filesets` for datasets.
- If the user wants to train on their own data, point to `customizations`.

## Acting on request

- Use `nemo_api` with the `models` resource for model entities and
  `files.filesets` for datasets, passing
  `workspace="<active request workspace>"` on every call.
- Use the exact entity name the user gave. Keep JSON `params` valid and
  compact.
- After a create or update, retrieve the entity and confirm the fields the
  user asked for before reporting success.
