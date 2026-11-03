// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import {
  IN_PROGRESS_JOB_STATUSES,
  useInProgressCustomizations,
} from '@studio/hooks/useInProgressCustomizations';
import { server } from '@studio/mocks/node';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import type { FC, PropsWithChildren } from 'react';

const JOBS_URL = `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs`;

const activeJob = {
  id: 'j1',
  name: 'grumpy-purple-gorilla',
  workspace: 'ws',
  source: 'customization',
  status: 'active',
  created_at: '2026-10-13T00:00:00Z',
  spec: {
    model: 'default/qwen3-0.6b',
    output: { name: 'grumpy-purple-gorilla' },
    training: { finetuning_type: 'lora' },
  },
};

const jobsResponse = (jobs: unknown[]) => ({
  data: jobs,
  pagination: { page: 1, page_size: 50, total_results: jobs.length, total_pages: 1 },
});

const createWrapper = (): FC<PropsWithChildren> => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useInProgressCustomizations', () => {
  it('maps a running job to the row the table needs', async () => {
    server.use(http.get(JOBS_URL, () => HttpResponse.json(jobsResponse([activeJob]))));

    const { result } = renderHook(() => useInProgressCustomizations('ws'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.runs).toHaveLength(1));
    expect(result.current.runs[0]).toEqual({
      jobName: 'grumpy-purple-gorilla',
      outputName: 'grumpy-purple-gorilla',
      status: 'active',
      baseModel: 'default/qwen3-0.6b',
      finetuningType: 'lora',
      createdAt: '2026-10-13T00:00:00Z',
    });
  });

  it('asks for in-flight statuses with $in, the only form the API matches', async () => {
    let capturedUrl: URL | undefined;
    server.use(
      http.get(JOBS_URL, ({ request }) => {
        capturedUrl = new URL(request.url);
        return HttpResponse.json(jobsResponse([]));
      })
    );

    renderHook(() => useInProgressCustomizations('ws'), { wrapper: createWrapper() });

    await waitFor(() => expect(capturedUrl).toBeDefined());

    // A plain array serializes as repeated `filter[status]` params and matches nothing.
    const statuses = capturedUrl?.searchParams.getAll('filter[status][$in]') ?? [];
    expect(statuses).toContain('active');
    expect(statuses).toContain('pending');
    expect(statuses).not.toContain('completed');
    expect(statuses).not.toContain('error');
    expect(capturedUrl?.searchParams.get('filter[source]')).toBe('customization');
  });

  it('reads the base model from the unsloth spec shape, where it is an object', async () => {
    server.use(
      http.get(JOBS_URL, () =>
        HttpResponse.json(
          jobsResponse([{ ...activeJob, spec: { ...activeJob.spec, model: { name: 'ws/llama' } } }])
        )
      )
    );

    const { result } = renderHook(() => useInProgressCustomizations('ws'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.runs).toHaveLength(1));
    expect(result.current.runs[0].baseModel).toBe('ws/llama');
  });

  it('drops a run that declares no output name', async () => {
    server.use(
      http.get(JOBS_URL, () => HttpResponse.json(jobsResponse([{ ...activeJob, spec: {} }])))
    );

    const { result } = renderHook(() => useInProgressCustomizations('ws'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.runs).toEqual([]);
  });

  it('ignores a finetuning type the UI cannot format', async () => {
    server.use(
      http.get(JOBS_URL, () =>
        HttpResponse.json(
          jobsResponse([
            { ...activeJob, spec: { ...activeJob.spec, training: { finetuning_type: 'wat' } } },
          ])
        )
      )
    );

    const { result } = renderHook(() => useInProgressCustomizations('ws'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.runs).toHaveLength(1));
    expect(result.current.runs[0].finetuningType).toBeUndefined();
  });
});

describe('IN_PROGRESS_JOB_STATUSES', () => {
  it('includes queued jobs, which have not produced a model yet', () => {
    expect(IN_PROGRESS_JOB_STATUSES).toContain(HelixJobStatus.queued);
  });
});
