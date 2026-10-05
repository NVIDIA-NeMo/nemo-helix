// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EvaluateJob, HelixJobStatus } from '@nemo/sdk/generated/evaluator/schema';

/** Alias used by hooks that work with the v2 evaluator API. */
export type EvaluationJobV2 = EvaluateJob;

export const getEvaluationJobId = (job: EvaluateJob) => {
  return job.id;
};

export const getEvaluationJobName = (job: EvaluateJob) => {
  return job.name;
};

export const getEvaluationJobModel = (job: EvaluateJob): string | undefined => {
  const target = job.spec?.target;
  if (!target) return undefined;
  if ('name' in target && typeof target.name === 'string') return target.name;
  return undefined;
};

export const getEvaluationJobConfigRef = (job: EvaluateJob): string | undefined => {
  const spec = job.spec;
  if (!spec) return undefined;

  const metrics = spec.metrics;
  if (Array.isArray(metrics) && metrics.length > 0) {
    const first = metrics[0];
    if (typeof first === 'string') return first;
    if (first && typeof first === 'object' && 'name' in first)
      return (first as { name: string }).name;
  }

  return undefined;
};

export const getEvaluationJobCustomFields = (job: EvaluateJob) => {
  return Object.keys(job?.custom_fields || {}) || [];
};

export const isEvaluationJobCreated = (status: HelixJobStatus | undefined) => {
  return status === 'created';
};

export const isEvaluationJobSucceeded = (status: HelixJobStatus | undefined) => {
  return status === 'completed';
};

export const isEvaluationJobInProgress = (status: HelixJobStatus | undefined) => {
  return status === 'active';
};

export const getEvaluationJobTarget = (job: EvaluateJob) => {
  return job.spec?.target;
};
