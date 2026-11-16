// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import type { HelixJobResponse } from '@nemo/sdk/generated/platform/schema';

export const makeEvaluatorJob = (overrides: Partial<HelixJobResponse> = {}): HelixJobResponse => ({
  id: 'job-1',
  attempt_id: 'attempt-1',
  name: 'job-1',
  workspace: DEFAULT_WORKSPACE,
  source: 'nemo-evals',
  spec: {
    target: { kind: 'fabric', source: { agent: `${DEFAULT_WORKSPACE}/my-agent` } },
    publication: { intake: { evaluation_id: 'published-eval' } },
  },
  platform_spec: { steps: [] },
  fileset: 'job-artifacts',
  status: 'completed',
  created_at: '2026-05-05T00:00:00Z',
  ...overrides,
});
