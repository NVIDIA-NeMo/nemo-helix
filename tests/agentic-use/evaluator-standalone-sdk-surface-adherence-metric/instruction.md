<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Add A Custom Surface-Adherence Metric

Use the local standalone Evals SDK to draft and exercise a custom metric that scores whether observed Evaluator surfaces stay within a task's allowed surfaces.

Use only `packages/nhx_evals_sdk` and SDK-level APIs in your answer. Do not propose the `nemo` CLI, plugin SDK APIs, or any `services/*` implementation path.

Your final answer must include:

- The package/path `packages/nhx_evals_sdk`.
- A directly runnable Python snippet defining a zero-argument metric class compatible with the SDK `Metric` protocol, including a `type` property.
- The API symbols `compute_scores`, `score_names`, `MetricResult`, and `MetricScore`.
- Score names `surface_adherence` and `surface_violation_count`.
- Logic that reads `observed_surfaces`, `allowed_surfaces`, and `forbidden_surfaces` from the dataset item.
- Treat `observed_surfaces` as the integration surfaces the evaluated path actually used, `allowed_surfaces` as the surfaces permitted for the task, and `forbidden_surfaces` as surfaces that should always count as violations.
- Define `surface_adherence` as a numeric score where `1.0` means all observed surfaces are allowed and no forbidden surfaces were used, and lower values indicate violations. Define `surface_violation_count` as the number of observed surfaces that are forbidden or outside the allowed set.
- A passing example where `observed_surfaces=["standalone_sdk"]` and `allowed_surfaces=["standalone_sdk"]` produces `surface_adherence=1.0` and `surface_violation_count=0`.
- A failing example where `observed_surfaces` includes `legacy_service` and `forbidden_surfaces=["legacy_service"]` produces `surface_adherence=0.0` or another penalty below `1.0`, with `surface_violation_count` greater than `0`.
