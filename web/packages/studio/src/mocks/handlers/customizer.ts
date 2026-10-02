// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { http, HttpResponse } from 'msw';

/**
 * Customizer detail-page handlers. Reads go through the GENERIC platform jobs API
 * (`/apis/jobs/v2/.../jobs/:name`); cancel goes to the per-backend customization collection.
 */
export const customizerHandlers = [
  // Empty by default; a suite that needs templates overrides this with server.use().
  http.get(`${PLATFORM_BASE_URL}/apis/customization/v2/workspaces/:workspace/job-templates`, () =>
    HttpResponse.json({ data: [], object: 'list' })
  ),
  // Generic single-job read used by the customization details page (useJobsGetJob).
  // Only answer for KNOWN customization jobs; fall through for any other job so this
  // handler doesn't shadow the generic JobDetailRoute (which also uses this endpoint).
  http.get(
    `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs/:name`,
    async ({ params }) => {
      const { customizationJobs } = await import('@studio/mocks/customizer/customization-jobs');
      const job = customizationJobs.find((candidate) => candidate.name === params.name);
      return job ? HttpResponse.json(job) : undefined;
    }
  ),
  http.get(
    `${PLATFORM_BASE_URL}/apis/customization/v2/workspaces/:workspace/:backend/jobs/:name/status`,
    async ({ params }) => {
      const { customizationJobSteps, failedGrpoCustomizationJob, failedGrpoJobSteps } =
        await import('@studio/mocks/customizer/customization-jobs');
      const hasFailed = params.name === failedGrpoCustomizationJob.name;
      return HttpResponse.json({
        id: 'job-status',
        name: params.name,
        status: hasFailed ? 'error' : 'completed',
        status_details: {},
        // Mirrors the backend: the job copies the failing step's generic text, and the mapped
        // cause stays down on the task.
        error_details: hasFailed ? { message: 'One or more tasks are in error state' } : {},
        steps: hasFailed ? failedGrpoJobSteps : customizationJobSteps,
        created_at: '2025-06-25T21:41:02.067430',
        updated_at: '2025-06-25T21:41:02.147000',
      });
    }
  ),
  // Per-backend cancel.
  http.post(
    `${PLATFORM_BASE_URL}/apis/customization/v2/workspaces/:workspace/:backend/jobs/:name/cancel`,
    async () => {
      const { customizationJob1 } = await import('@studio/mocks/customizer/customization-jobs');
      return HttpResponse.json(customizationJob1);
    }
  ),
];
