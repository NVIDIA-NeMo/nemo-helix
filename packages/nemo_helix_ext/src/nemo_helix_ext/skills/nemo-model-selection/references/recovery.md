<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Recovery when verification fails

| Symptom | Cause | Recovery |
|---|---|---|
| User says "wrong recommendation" | One of the three profile answers was misread | Re-ask just that question; do not restart |
| User wants a model not in the table | The table is curated, not exhaustive | Tell them honestly; describe the capability gap their choice would have vs the closest recommended model |
| `cache_missing` and user wants fresh data | Cache has never been refreshed in this checkout | Tell them the refresh command and note that the static table is still usable |
| User picks self-hosted but the recommended model needs cloud | Hard constraint conflict | Drop the recommendation; pick the closest self-hosted-compatible model from the table |
| Available model's harness API support is unknown | Availability was mistaken for harness compatibility | Ask permission to run a valid request through the harness's required model path |
| Compatibility request fails | The exact harness/provider/model/endpoint combination is incompatible in the current environment | Exclude that combination, surface the error, and test the next ranked candidate after the user-approved probe set |
| No candidate passes | No live model satisfies the selected harness contract | Stop without emitting a model block; ask the user to configure a compatible provider or explicitly choose another harness |

