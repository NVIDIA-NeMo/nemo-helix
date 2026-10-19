// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { useCustomizationJobForModel } from '@studio/hooks/useCustomizationJobForModel';
import { server } from '@studio/mocks/node';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import type { FC, PropsWithChildren } from 'react';

const JOBS_URL = `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs`;

const buildModel = (overrides: Partial<ModelEntity> = {}): ModelEntity =>
  ({
    id: 'model-1',
    name: 'my-checkpoint',
    workspace: 'ws',
    fileset: 'ws/my-checkpoint',
    finetuning_type: 'all_weights',
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T00:00:00Z',
    ...overrides,
  }) as ModelEntity;

const buildAdapter = (overrides: Partial<Adapter> = {}): Adapter =>
  ({
    name: 'grumpy-purple-gorilla',
    workspace: 'ws',
    fileset: 'ws/grumpy-purple-gorilla',
    finetuning_type: 'lora',
    created_at: '2025-01-02T00:00:00Z',
    ...overrides,
  }) as Adapter;

const jobsResponse = (jobs: { id: string; name: string }[]) => ({
  data: jobs,
  pagination: { page: 1, page_size: 1, total_results: jobs.length, total_pages: 1 },
});

const createWrapper = (): FC<PropsWithChildren> => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useCustomizationJobForModel', () => {
  it('queries jobs by source and exact spec.output.name and returns the job name', async () => {
    let capturedUrl: URL | undefined;
    server.use(
      http.get(JOBS_URL, ({ request }) => {
        capturedUrl = new URL(request.url);
        return HttpResponse.json(jobsResponse([{ id: 'j1', name: 'automodel-2a4c86bf' }]));
      })
    );

    const { result } = renderHook(() => useCustomizationJobForModel(buildModel()), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.jobName).toBe('automodel-2a4c86bf'));

    expect(capturedUrl?.searchParams.get('filter[source]')).toBe('customization');
    expect(capturedUrl?.searchParams.get('filter[spec.output.name]')).toBe('my-checkpoint');
  });

  it('does not query when no model is selected', async () => {
    const handler = vi.fn(() => HttpResponse.json(jobsResponse([])));
    server.use(http.get(JOBS_URL, handler));

    const { result } = renderHook(() => useCustomizationJobForModel(null), {
      wrapper: createWrapper(),
    });

    expect(result.current.jobName).toBeUndefined();
    expect(result.current.isLoading).toBe(false);
    expect(handler).not.toHaveBeenCalled();
  });

  it('returns undefined when no customization job produced the model', async () => {
    server.use(http.get(JOBS_URL, () => HttpResponse.json(jobsResponse([]))));

    const { result } = renderHook(() => useCustomizationJobForModel(buildModel()), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.jobName).toBeUndefined();
  });

  it("queries the entity's own workspace, not the one the UI is showing", async () => {
    let capturedWorkspace: string | undefined;
    server.use(
      http.get(JOBS_URL, ({ params }) => {
        capturedWorkspace = params.workspace as string;
        return HttpResponse.json(jobsResponse([{ id: 'j1', name: 'automodel-2a4c86bf' }]));
      })
    );

    const { result } = renderHook(
      () => useCustomizationJobForModel(buildModel({ workspace: 'other-ws' })),
      { wrapper: createWrapper() }
    );

    await waitFor(() => expect(result.current.jobName).toBe('automodel-2a4c86bf'));
    expect(capturedWorkspace).toBe('other-ws');
  });

  describe('when given an adapter', () => {
    it("resolves the run by the adapter's name, in the adapter's workspace", async () => {
      let capturedUrl: URL | undefined;
      let capturedWorkspace: string | undefined;
      server.use(
        http.get(JOBS_URL, ({ request, params }) => {
          capturedUrl = new URL(request.url);
          capturedWorkspace = params.workspace as string;
          return HttpResponse.json(jobsResponse([{ id: 'j2', name: 'lora-job-456' }]));
        })
      );

      const { result } = renderHook(
        () => useCustomizationJobForModel(buildAdapter({ workspace: 'adapter-ws' })),
        { wrapper: createWrapper() }
      );

      await waitFor(() => expect(result.current.jobName).toBe('lora-job-456'));

      // The lookup key is the adapter's name -- a LoRA run writes that to
      // spec.output.name -- and is distinct from the name of the job it finds.
      expect(capturedUrl?.searchParams.get('filter[spec.output.name]')).toBe(
        'grumpy-purple-gorilla'
      );
      // An adapter may live in a different workspace from its base; the job lives where
      // its output was registered.
      expect(capturedWorkspace).toBe('adapter-ws');
    });

    it('does not query when the adapter has no name', async () => {
      const handler = vi.fn(() => HttpResponse.json(jobsResponse([])));
      server.use(http.get(JOBS_URL, handler));

      const { result } = renderHook(() => useCustomizationJobForModel(buildAdapter({ name: '' })), {
        wrapper: createWrapper(),
      });

      expect(result.current.jobName).toBeUndefined();
      expect(result.current.isLoading).toBe(false);
      expect(handler).not.toHaveBeenCalled();
    });

    it('does not query when the entity has no workspace', async () => {
      const handler = vi.fn(() => HttpResponse.json(jobsResponse([])));
      server.use(http.get(JOBS_URL, handler));

      const { result } = renderHook(
        () => useCustomizationJobForModel(buildAdapter({ workspace: '' })),
        { wrapper: createWrapper() }
      );

      expect(result.current.jobName).toBeUndefined();
      expect(handler).not.toHaveBeenCalled();
    });
  });
});
