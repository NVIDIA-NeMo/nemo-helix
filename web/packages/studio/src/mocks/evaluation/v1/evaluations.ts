// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type EvaluateJob,
  type EvaluateJobsPage,
  HelixJobStatus,
} from '@nemo/sdk/generated/evaluator/schema';

export const mockEvalConfigOnline1 = {
  id: 'eval-config-online-1',
  namespace: 'default',
  name: 'online-eval-config-1',
  custom_fields: {},
  type: 'custom',
  schema_version: '1.0',
  version_id: '',
  tasks: {
    default: {
      type: 'chat-completion',
      dataset: {
        files_url: 'hf://datasets/e2e-namespace/e2e_eval_fileset/eval.json',
      },
      params: {
        batch_size: 1,
        max_length_generation: 512,
        temperature: 1.0,
        top_k: 1,
        top_p: 0.0,
        n_samples: 1,
        num_chunks: 1,
      },
      metrics: { bleu: { type: 'bleu' } },
    },
  },
  params: {},
};

export const mockEvalConfigOffline1 = {
  id: 'eval-config-offline-1',
  namespace: 'default',
  name: 'offline-eval-config-1',
  custom_fields: {},
  type: 'custom',
  schema_version: '1.0',
  type_prefix: null,
  version_id: '',
  tasks: {
    default: {
      type: 'data',
      dataset: {
        files_url: 'input.json',
      },
      params: {
        batch_size: 1,
        max_length_generation: 512,
        temperature: 1.0,
        top_k: 1,
        top_p: 0.0,
        n_samples: 1,
        num_chunks: 1,
      },
      metrics: { bleu: { type: 'bleu' } },
    },
  },
  params: {},
};

export const getEvaluationConfigsListResponse: EvaluateJobsPage = {
  pagination: {
    total_results: 2,
    total_pages: 1,
    current_page_size: 2,
    page: 1,
    page_size: 2,
  },
  data: [mockEvalConfigOnline1, mockEvalConfigOffline1] as unknown as EvaluateJob[],
};

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

export const metricEvaluationJob2: EvaluateJob = {
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

export const metricEvaluationJob3: EvaluateJob = {
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
