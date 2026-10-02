---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: entities
description: NeMo Helix model and dataset CRUD lifecycle through the platform SDK, with strict field matching on final verification entities.
---
Entity tasks

- Use `nemo_api` with the `models` or `datasets` resource as appropriate, passing
  `workspace="<active request workspace>"` on every call.
- Use the exact entity type and name from the instruction.
- For model/dataset CRUD, create the temporary entity, verify/list/update/delete it, then create the final verification entity.
- Keep JSON params valid and compact. Final dataset checks often require fields such as `format` and `size` to match exactly.
- Retrieve or list the final model or dataset and compare every required field,
  including its exact entity type and name, before reporting success.
