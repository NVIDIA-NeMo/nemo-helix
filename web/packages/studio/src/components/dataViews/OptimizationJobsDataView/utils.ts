// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Trial } from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';

export const EM_DASH = '—';

/** Sort id prefix for the one-column-per-objective metric columns. */
export const METRIC_SORT_PREFIX = 'metric:';

/** Sort id prefix for the one-column-per-parameter columns. */
export const PARAM_SORT_PREFIX = 'param:';

export const formatParamValue = (value: string): string => {
  const parsed = Number(value);
  if (value.trim() === '' || Number.isNaN(parsed) || Number.isInteger(parsed)) return value;
  const rounded = Number(parsed.toFixed(3));
  return rounded === 0 ? value : String(rounded);
};

export const paramValue = (trial: Trial, name: string): string | undefined =>
  trial.params.find((param) => param.name === name)?.value;
