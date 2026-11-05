// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CommitOutput } from '@huggingface/hub';
import type { FilesetOutput, FilesetOutputsPage } from '@nemo/sdk/generated/platform/schema';

const mockFileset = (id: string, name: string, workspace: string): FilesetOutput => ({
  id,
  name,
  workspace,
  description: '',
  purpose: 'dataset',
  storage: { type: 'local', path: 'local/path' },
  metadata: {},
  custom_fields: {},
  project: workspace,
  created_at: '2024-12-17T16:08:56.880768',
  updated_at: '2024-12-17T16:08:56.880771',
});

export const datasets: FilesetOutputsPage = {
  data: [
    mockFileset('dataset-EabKMKEAZpxqBruYU9Yfhf', 'dataset-187625', 'default'),
    mockFileset('dataset-EabKMKEAZpxqBruYU9Yfh2', 'dataset-337067', 'default'),
    mockFileset('dataset-EabKMKEAZpxqBruYU9Yfh3', 'dataset-869340', 'default'),
    mockFileset('dataset-EabKMKEAZpxqBruYU9Yfh4', 'dataset-358646', 'default'),
    mockFileset('dataset-EabKMKEAZpxqBruYU9Yfh5', 'dataset-657702', 'default'),
  ],
  pagination: {
    page: 1,
    page_size: 50,
    current_page_size: 49,
    total_pages: 1,
    total_results: 49,
  },
  sort: '-created_at',
};

export const dataset = datasets.data[0] as FilesetOutput;

export const createDatasetRepoResponse = {
  repoUrl: 'datasets/default/dataset-657702',
};

export const uploadFileResponse: CommitOutput = {
  commit: {
    oid: '2a1b1cdf9c45a21b1daf606f1800a3edf20cc003d5af402fd6d712c5a9df6c17',
    url: 'hf://',
  },
  hookOutput: '',
};

export const preuploadFilesResponse = {
  files: [
    {
      path: 'training_file.jsonl',
      uploadMode: 'lfs',
      shouldIgnore: false,
    },
    {
      path: 'validation_file.jsonl',
      uploadMode: 'lfs',
      shouldIgnore: false,
    },
  ],
};

export const uploadFilesBatchResponse = {
  objects: [
    {
      oid: 'cfa2fd49abe343ad5d02f88f36592084b8c6e39db116977ce57f8baae9f3316b',
      size: 3817191,
    },
    {
      oid: '2a1b1cdf9c45a21b1daf606f1800a3edf20cc003d5af402fd6d712c5a9df6c18',
      size: 499572,
    },
  ],
};

export const commitResponse = {
  success: true,
  commitOid: '6fcc0f7e943059d59413d7b5bd5aa4855ef5928c',
  commitUrl: '',
};

export const mockLfsObjectResponse = JSON.stringify({ key: 'value' });

/**
 * Mock filesets specifically for testing bulk delete operations.
 * These filesets have consistent naming and workspace for predictable testing.
 */
export const bulkDeleteTestDatasets: FilesetOutput[] = [
  mockFileset('uuid-1', 'dataset-1', 'test-namespace'),
  mockFileset('uuid-2', 'dataset-2', 'test-namespace'),
  mockFileset('uuid-3', 'dataset-3', 'test-namespace'),
];

/**
 * Mock fileset with undefined workspace for edge case testing.
 */
export const datasetWithUndefinedNamespace: FilesetOutput = {
  id: 'uuid-undefined-ns',
  name: 'dataset-undefined-ns',
  workspace: undefined as unknown as string,
  description: '',
  purpose: 'dataset',
  storage: { type: 'local', path: 'local/path' },
  metadata: {},
  custom_fields: {},
  project: 'default',
  created_at: '2024-12-17T16:08:56.880768',
  updated_at: '2024-12-17T16:08:56.880771',
};

/**
 * Mock fileset with undefined name for edge case testing.
 */
export const datasetWithUndefinedName: FilesetOutput = {
  id: 'uuid-undefined-name',
  name: undefined as unknown as string,
  workspace: 'test-namespace',
  description: '',
  purpose: 'dataset',
  storage: { type: 'local', path: 'local/path' },
  metadata: {},
  custom_fields: {},
  project: 'test-namespace',
  created_at: '2024-12-17T16:08:56.880768',
  updated_at: '2024-12-17T16:08:56.880771',
};
