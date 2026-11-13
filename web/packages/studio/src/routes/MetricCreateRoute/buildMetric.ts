// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { MetricInline } from '@nemo/sdk/generated/evals/schema';
import { buildMetricBundle } from '@studio/components/evaluation/EvalWizard/buildEvaluationSpec';
import { CANONICAL_BINDINGS, type MetricFormValues } from '@studio/routes/MetricCreateRoute/types';

/** Form values -> the metric bundle the API stores. The user's description, when
 *  given, replaces the generated one. */
export const buildMetric = (values: MetricFormValues, workspace: string): MetricInline => {
  const bundle = buildMetricBundle(values.type, values.body, CANONICAL_BINDINGS, workspace);
  // Canonical bindings always carry a reference, so every type builds.
  if (!bundle) throw new Error(`Could not build a ${values.type} metric.`);
  const description = values.description.trim();
  return {
    ...bundle,
    metadata: { ...bundle.metadata, ...(description ? { description } : {}) },
  } as unknown as MetricInline;
};
