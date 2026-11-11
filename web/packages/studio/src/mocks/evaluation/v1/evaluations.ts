// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EvaluateJob, EvaluateJobsPage } from '@nemo/sdk/generated/evals/schema';
import { HelixJobStatus } from '@nemo/sdk/generated/evals/schema/HelixJobStatus';

export const metricEvaluationJob1: EvaluateJob = {
  id: 'eval-job-1',
  name: 'eval-job-1',
  workspace: 'default',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T01:00:00Z',
  status: HelixJobStatus.completed,
  spec: {
    metrics: [],
    dataset: `default/test-dataset` as unknown as EvaluateJob['spec']['dataset'],
  },
  custom_fields: {},
};

const metricEvaluationJob2: EvaluateJob = {
  id: 'eval-job-2',
  name: 'eval-job-2',
  workspace: 'default',
  created_at: '2024-01-02T00:00:00Z',
  updated_at: '2024-01-02T01:00:00Z',
  status: HelixJobStatus.completed,
  spec: {
    metrics: [],
    dataset: `default/test-dataset-2` as unknown as EvaluateJob['spec']['dataset'],
  },
  custom_fields: {},
};

const metricEvaluationJob3: EvaluateJob = {
  id: 'eval-job-3',
  name: 'eval-job-3',
  workspace: 'default',
  created_at: '2024-01-03T00:00:00Z',
  updated_at: '2024-01-03T01:00:00Z',
  status: HelixJobStatus.completed,
  spec: {
    metrics: [],
    dataset: `default/test-dataset-3` as unknown as EvaluateJob['spec']['dataset'],
  },
  custom_fields: {},
};

export const metricEvaluationJobsPage: EvaluateJobsPage = {
  data: [metricEvaluationJob1, metricEvaluationJob2, metricEvaluationJob3],
  pagination: {
    total_results: 3,
    total_pages: 1,
    current_page_size: 3,
    page: 1,
    page_size: 25,
  },
};
