// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { useCustomizationJobsForModels } from '@studio/hooks/useCustomizationJobsForModels';
import { server } from '@studio/mocks/node';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import type { FC, PropsWithChildren } from 'react';

const JOBS_URL = `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs`;

const job = (name: string, outputName: string, createdAt: string) => ({
  id: `id-${name}`,
  name,
  workspace: 'ws',
  source: 'customization',
  status: 'completed',
  created_at: createdAt,
  spec: { output: { name: outputName } },
});

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

describe('useCustomizationJobsForModels', () => {
  it('maps every output name to its originating job in one request', async () => {
    let requestCount = 0;
    server.use(
      http.get(JOBS_URL, () => {
        requestCount += 1;
        return HttpResponse.json(
          jobsResponse([
            job('grumpy-purple-gorilla', 'grumpy-purple-gorilla', '2026-10-13T00:00:00Z'),
            job('automodel-abc', 'my-checkpoint', '2026-10-12T00:00:00Z'),
          ])
        );
      })
    );

    const { result } = renderHook(
      () => useCustomizationJobsForModels('ws', ['grumpy-purple-gorilla', 'my-checkpoint']),
      { wrapper: createWrapper() }
    );

    await waitFor(() => expect(result.current.jobsByOutputName.size).toBe(2));
    expect(result.current.jobsByOutputName.get('grumpy-purple-gorilla')).toBe(
      'grumpy-purple-gorilla'
    );
    expect(result.current.jobsByOutputName.get('my-checkpoint')).toBe('automodel-abc');
    expect(requestCount).toBe(1);
  });

  it('sends the names as an $in filter', async () => {
    let capturedUrl: URL | undefined;
    server.use(
      http.get(JOBS_URL, ({ request }) => {
        capturedUrl = new URL(request.url);
        return HttpResponse.json(jobsResponse([]));
      })
    );

    renderHook(() => useCustomizationJobsForModels('ws', ['b-model', 'a-model']), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(capturedUrl).toBeDefined());

    // The plain-array form serializes as repeated `filter[spec.output.name]` params, which
    // the API matches zero jobs against. Only the `$in` operator form filters correctly.
    const names = capturedUrl?.searchParams.getAll('filter[spec.output.name][$in]') ?? [];
    expect(names).toEqual(['a-model', 'b-model']);
    expect(capturedUrl?.searchParams.get('filter[source]')).toBe('customization');
  });

  it('keeps the newest job when a name was reused across re-runs', async () => {
    server.use(
      http.get(JOBS_URL, () =>
        HttpResponse.json(
          jobsResponse([
            job('run-newer', 'my-checkpoint', '2026-10-13T00:00:00Z'),
            job('run-older', 'my-checkpoint', '2026-10-01T00:00:00Z'),
          ])
        )
      )
    );

    const { result } = renderHook(() => useCustomizationJobsForModels('ws', ['my-checkpoint']), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.jobsByOutputName.size).toBe(1));
    expect(result.current.jobsByOutputName.get('my-checkpoint')).toBe('run-newer');
  });

  it('stays idle when there are no names to resolve', async () => {
    const handler = vi.fn(() => HttpResponse.json(jobsResponse([])));
    server.use(http.get(JOBS_URL, handler));

    const { result } = renderHook(() => useCustomizationJobsForModels('ws', []), {
      wrapper: createWrapper(),
    });

    expect(result.current.jobsByOutputName.size).toBe(0);
    expect(result.current.isLoading).toBe(false);
    expect(handler).not.toHaveBeenCalled();
  });
});
