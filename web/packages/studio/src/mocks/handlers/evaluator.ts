// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { http, HttpResponse } from 'msw';

export const evaluatorHandlers = [
  // nemo-evals agent-evaluate jobs — Studio's agent-evaluation feature runs
  // here. Default to an empty list / 404 so the list + detail routes render
  // their empty/not-found states in tests without per-test overrides.
  http.get(`${PLATFORM_BASE_URL}/apis/evals/v2/workspaces/:workspace/agent-evaluate/jobs`, () =>
    HttpResponse.json({ data: [], pagination: { total: 0, page: 1, page_size: 50 } })
  ),
  http.get(
    `${PLATFORM_BASE_URL}/apis/evals/v2/workspaces/:workspace/agent-evaluate/jobs/:name`,
    () => HttpResponse.json({ detail: 'Not found' }, { status: 404 })
  ),
  http.get(
    `${PLATFORM_BASE_URL}/apis/evals/v2/workspaces/:workspace/agent-evaluate/jobs/:name/status`,
    () => HttpResponse.json({ name: '', status: 'unknown' })
  ),
  http.get(
    `${PLATFORM_BASE_URL}/apis/evals/v2/workspaces/:workspace/agent-eval-results/:name`,
    () => HttpResponse.json({ detail: 'Not found' }, { status: 404 })
  ),
];
