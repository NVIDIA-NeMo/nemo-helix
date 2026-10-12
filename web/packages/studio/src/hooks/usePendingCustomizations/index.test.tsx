// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import {
  getPendingJobName,
  isPendingModelEntity,
  toPendingModelEntity,
  usePendingCustomizations,
  type PendingCustomization,
} from '@studio/hooks/usePendingCustomizations';
import { server } from '@studio/mocks/node';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import type { FC, PropsWithChildren } from 'react';

const JOBS_URL = `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs`;

const jobsResponse = (jobs: unknown[]) => ({
  data: jobs,
  pagination: { page: 1, page_size: 50, total_results: jobs.length, total_pages: 1 },
});

const automodelJob = {
  id: 'j1',
  name: 'automodel-003',
  workspace: 'ws',
  source: 'customization',
  status: 'active',
  created_at: '2026-10-06T00:00:00Z',
  spec: { model: 'default/nemotron-3.5-lightning', output: { name: 'bird-sql-003' } },
};

const createWrapper = (): FC<PropsWithChildren> => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('usePendingCustomizations', () => {
  it('queries in-flight customization jobs and exposes their declared output model', async () => {
    let capturedUrl: URL | undefined;
    server.use(
      http.get(JOBS_URL, ({ request }) => {
        capturedUrl = new URL(request.url);
        return HttpResponse.json(jobsResponse([automodelJob]));
      })
    );

    const { result } = renderHook(() => usePendingCustomizations('ws'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.pending).toHaveLength(1));

    expect(result.current.pending[0]).toEqual({
      jobName: 'automodel-003',
      outputModelName: 'bird-sql-003',
      status: 'active',
      baseModel: 'default/nemotron-3.5-lightning',
      createdAt: '2026-10-06T00:00:00Z',
      workspace: 'ws',
    });
    expect(capturedUrl?.searchParams.get('filter[source]')).toBe('customization');
    // Terminal states are excluded: those runs have registered their model already.
    const statuses = capturedUrl?.searchParams.getAll('filter[status]') ?? [];
    expect(statuses.join(',')).toContain('active');
    expect(statuses.join(',')).not.toContain('completed');
  });

  it('reads the base model from the unsloth spec shape, where it is an object', async () => {
    server.use(
      http.get(JOBS_URL, () =>
        HttpResponse.json(
          jobsResponse([
            { ...automodelJob, spec: { model: { name: 'ws/llama' }, output: { name: 'out' } } },
          ])
        )
      )
    );

    const { result } = renderHook(() => usePendingCustomizations('ws'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.pending).toHaveLength(1));
    expect(result.current.pending[0].baseModel).toBe('ws/llama');
  });

  it('drops runs that declare no output name, having no model to stand in for', async () => {
    server.use(
      http.get(JOBS_URL, () => HttpResponse.json(jobsResponse([{ ...automodelJob, spec: {} }])))
    );

    const { result } = renderHook(() => usePendingCustomizations('ws'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.pending).toEqual([]);
  });

  it('stays idle without a workspace', async () => {
    const handler = vi.fn(() => HttpResponse.json(jobsResponse([])));
    server.use(http.get(JOBS_URL, handler));

    const { result } = renderHook(() => usePendingCustomizations(null), {
      wrapper: createWrapper(),
    });

    expect(result.current.pending).toEqual([]);
    expect(result.current.isLoading).toBe(false);
    expect(handler).not.toHaveBeenCalled();
  });
});

describe('toPendingModelEntity', () => {
  const pending: PendingCustomization = {
    jobName: 'automodel-003',
    outputModelName: 'bird-sql-003',
    status: 'active',
    baseModel: 'default/nemotron',
    createdAt: '2026-10-06T00:00:00Z',
    workspace: 'ws',
  };

  it('round-trips the originating job name through the synthetic entity', () => {
    const entity = toPendingModelEntity(pending);

    expect(entity.name).toBe('bird-sql-003');
    expect(entity.workspace).toBe('ws');
    expect(entity.base_model).toBe('default/nemotron');
    expect(isPendingModelEntity(entity)).toBe(true);
    expect(getPendingJobName(entity)).toBe('automodel-003');
  });

  it('does not claim a real model entity', () => {
    expect(isPendingModelEntity({ id: 'd4b5e1a0-0000-4000-8000-000000000000' })).toBe(false);
    expect(getPendingJobName({ id: 'd4b5e1a0-0000-4000-8000-000000000000' })).toBeUndefined();
  });
});
