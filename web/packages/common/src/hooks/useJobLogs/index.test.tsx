// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { jobsPageJobLogs } from '@nemo/sdk/generated/platform/jobs';
import type { HelixJobLog, HelixJobLogPage } from '@nemo/sdk/generated/platform/schema';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';

import { useJobLogs } from './index';

vi.mock('@nemo/sdk/generated/platform/jobs', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@nemo/sdk/generated/platform/jobs')>();
  return {
    ...actual,
    jobsPageJobLogs: vi.fn(),
  };
});

const mockJobsPageJobLogs = vi.mocked(jobsPageJobLogs);

const WORKSPACE = 'test-workspace';
const JOB_NAME = 'test-job';

function makeLog(index: number): HelixJobLog {
  return {
    timestamp: `2026-01-01T00:00:${String(index).padStart(2, '0')}Z`,
    job: JOB_NAME,
    job_step: 'step',
    job_task: 'task',
    message: `Log message ${index}`,
  };
}

function makePage(
  logs: HelixJobLog[],
  total: number,
  nextPage: string | null = null
): HelixJobLogPage {
  return {
    data: logs,
    total,
    next_page: nextPage ?? '',
    prev_page: '',
  };
}

/** Drains every pending microtask, so a queryFn's page walk runs to completion. */
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe('useJobLogs', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns logs from a single page', async () => {
    const logs = [makeLog(0), makeLog(1), makeLog(2)];
    mockJobsPageJobLogs.mockResolvedValueOnce(makePage(logs, 3));

    const { result } = renderHook(() => useJobLogs({ workspace: WORKSPACE, name: JOB_NAME }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.data).toEqual(logs);
    expect(result.current.total).toBe(3);
    expect(result.current.error).toBeNull();
  });

  it('paginates through multiple pages', async () => {
    const page1 = Array.from({ length: 3 }, (_, i) => makeLog(i));
    const page2 = Array.from({ length: 2 }, (_, i) => makeLog(i + 3));

    mockJobsPageJobLogs
      .mockResolvedValueOnce(makePage(page1, 5, 'cursor-1'))
      .mockResolvedValueOnce(makePage(page2, 5));

    const { result } = renderHook(
      () => useJobLogs({ workspace: WORKSPACE, name: JOB_NAME, pageSize: 3 }),
      { wrapper: createWrapper() }
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.data).toEqual([...page1, ...page2]);
    expect(result.current.total).toBe(5);
    expect(mockJobsPageJobLogs).toHaveBeenCalledTimes(2);
  });

  it('returns empty array when there are no logs', async () => {
    mockJobsPageJobLogs.mockResolvedValueOnce(makePage([], 0));

    const { result } = renderHook(() => useJobLogs({ workspace: WORKSPACE, name: JOB_NAME }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.data).toEqual([]);
    expect(result.current.total).toBe(0);
  });

  it('is disabled when workspace or name is empty', () => {
    const { result } = renderHook(() => useJobLogs({ workspace: '', name: '' }), {
      wrapper: createWrapper(),
    });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toEqual([]);
    expect(mockJobsPageJobLogs).not.toHaveBeenCalled();
  });

  it('respects explicit enabled: false', () => {
    const { result } = renderHook(
      () => useJobLogs({ workspace: WORKSPACE, name: JOB_NAME, enabled: false }),
      { wrapper: createWrapper() }
    );

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toEqual([]);
    expect(mockJobsPageJobLogs).not.toHaveBeenCalled();
  });

  it('reports errors', async () => {
    mockJobsPageJobLogs.mockRejectedValueOnce(new Error('Network failure'));

    const { result } = renderHook(() => useJobLogs({ workspace: WORKSPACE, name: JOB_NAME }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.error).toBeTruthy());
  });

  it('trims logs to maxRetainedLogs keeping the tail', async () => {
    // maxPages=5, pageSize=2 -> maxRetainedLogs=10
    // 6 full pages of 2 = 12 logs total, should trim to last 10
    const pageSize = 2;
    const maxPages = 5;
    const maxRetained = maxPages * pageSize; // 10

    const pages = Array.from({ length: 6 }, (_, p) =>
      Array.from({ length: 2 }, (_, i) => makeLog(p * 2 + i))
    );

    mockJobsPageJobLogs
      .mockResolvedValueOnce(makePage(pages[0], 12, 'c1'))
      .mockResolvedValueOnce(makePage(pages[1], 12, 'c2'))
      .mockResolvedValueOnce(makePage(pages[2], 12, 'c3'))
      .mockResolvedValueOnce(makePage(pages[3], 12, 'c4'))
      .mockResolvedValueOnce(makePage(pages[4], 12, 'c5'))
      .mockResolvedValueOnce(makePage(pages[5], 12));

    const { result } = renderHook(
      () => useJobLogs({ workspace: WORKSPACE, name: JOB_NAME, pageSize, maxPages }),
      { wrapper: createWrapper() }
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.data).toHaveLength(maxRetained);
    expect(result.current.data[0].message).toBe('Log message 2');
    expect(result.current.data[9].message).toBe('Log message 11');
  });

  it('caches full pages and only refetches the last page', async () => {
    const pageSize = 3;
    const fullPage = Array.from({ length: 3 }, (_, i) => makeLog(i));
    const lastPage = [makeLog(3)];
    const lastPageUpdated = [makeLog(3), makeLog(4)];

    // Initial fetch: full page + partial last page
    mockJobsPageJobLogs
      .mockResolvedValueOnce(makePage(fullPage, 4, 'c1'))
      .mockResolvedValueOnce(makePage(lastPage, 4));

    // Per-page cache entries need gcTime > 0 to survive between outer query runs
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false, gcTime: Infinity },
      },
    });
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    const { result } = renderHook(
      () => useJobLogs({ workspace: WORKSPACE, name: JOB_NAME, pageSize }),
      { wrapper }
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.data).toHaveLength(4);
    expect(mockJobsPageJobLogs).toHaveBeenCalledTimes(2);

    // Simulate refetch: full page should come from cache, only last page refetches
    mockJobsPageJobLogs.mockClear();
    mockJobsPageJobLogs.mockResolvedValueOnce(makePage(lastPageUpdated, 5));

    const { data: refetchData } = await act(() => result.current.refetch());

    // Only 1 API call (last page). Full page resolved from cache.
    expect(mockJobsPageJobLogs).toHaveBeenCalledTimes(1);
    expect(refetchData?.logs).toHaveLength(5);
  });

  it('fetches all logs on manual refetch when disabled', async () => {
    const logs = [makeLog(0), makeLog(1)];
    mockJobsPageJobLogs.mockResolvedValueOnce(makePage(logs, 2));

    const { result } = renderHook(
      () => useJobLogs({ workspace: WORKSPACE, name: JOB_NAME, enabled: false }),
      { wrapper: createWrapper() }
    );

    expect(mockJobsPageJobLogs).not.toHaveBeenCalled();

    const { data } = await act(() => result.current.refetch());

    expect(mockJobsPageJobLogs).toHaveBeenCalledTimes(1);
    expect(data?.logs).toEqual(logs);
  });
  describe('loadProgress', () => {
    // The page walk reports progress via setState from inside the queryFn, so each
    // step has to be flushed inside act() rather than awaited with waitFor.
    it('advances as the walk pages through the log', async () => {
      const page1 = deferred<HelixJobLogPage>();
      const page2 = deferred<HelixJobLogPage>();
      mockJobsPageJobLogs.mockReturnValueOnce(page1.promise).mockReturnValueOnce(page2.promise);

      const { result } = renderHook(
        () => useJobLogs({ workspace: WORKSPACE, name: JOB_NAME, pageSize: 2 }),
        { wrapper: createWrapper() }
      );

      // Nothing to report until the first page tells us the total.
      expect(result.current.loadProgress).toBeNull();

      await act(async () => {
        page1.resolve(makePage([makeLog(0), makeLog(1)], 4, 'c1'));
        await tick();
      });

      expect(result.current.loadProgress).toEqual({ loaded: 2, total: 4 });
      // The walk is still mid-flight, which is exactly when this matters.
      expect(result.current.isLoading).toBe(true);

      await act(async () => {
        page2.resolve(makePage([makeLog(2), makeLog(3)], 4));
        await tick();
      });

      await waitFor(() => expect(result.current.isLoading).toBe(false));
      expect(result.current.loadProgress).toEqual({ loaded: 4, total: 4 });
    });

    it('counts fetched lines rather than the trimmed retained set', async () => {
      // maxPages=5, pageSize=2 retains 10 lines, but the walk still fetches all 12.
      const pageSize = 2;
      const maxPages = 5;
      const pages = Array.from({ length: 6 }, (_, p) =>
        Array.from({ length: 2 }, (_, i) => makeLog(p * 2 + i))
      );
      pages.forEach((page, i) =>
        mockJobsPageJobLogs.mockResolvedValueOnce(
          makePage(page, 12, i === pages.length - 1 ? null : `c${i + 1}`)
        )
      );

      const { result } = renderHook(
        () => useJobLogs({ workspace: WORKSPACE, name: JOB_NAME, pageSize, maxPages }),
        { wrapper: createWrapper() }
      );

      await waitFor(() => expect(result.current.isLoading).toBe(false));

      expect(result.current.data).toHaveLength(10);
      expect(result.current.loadProgress).toEqual({ loaded: 12, total: 12 });
    });

    it('clears between walks so a refetch never shows a stale count', async () => {
      mockJobsPageJobLogs.mockResolvedValueOnce(makePage([makeLog(0), makeLog(1)], 2));

      const { result } = renderHook(
        () => useJobLogs({ workspace: WORKSPACE, name: JOB_NAME, pageSize: 2 }),
        { wrapper: createWrapper() }
      );

      await act(async () => {
        await tick();
      });
      expect(result.current.loadProgress).toEqual({ loaded: 2, total: 2 });

      const nextWalk = deferred<HelixJobLogPage>();
      mockJobsPageJobLogs.mockReturnValueOnce(nextWalk.promise);

      await act(async () => {
        void result.current.refetch();
        await tick();
      });
      expect(result.current.loadProgress).toBeNull();

      await act(async () => {
        nextWalk.resolve(makePage([makeLog(0), makeLog(1), makeLog(2)], 3));
        await tick();
      });
      expect(result.current.loadProgress).toEqual({ loaded: 3, total: 3 });
    });

    it("never carries a finished job's count into the next job", async () => {
      mockJobsPageJobLogs.mockResolvedValueOnce(makePage([makeLog(0), makeLog(1)], 2));

      // The route element is reused across :jobName changes, so the hook re-runs
      // without remounting. Record every rendered value, not just the settled one —
      // the regression this guards is a single frame of the old job's count.
      const seen: ({ loaded: number; total: number } | null)[] = [];
      const { result, rerender } = renderHook(
        ({ name }) => {
          const jobLogs = useJobLogs({ workspace: WORKSPACE, name, pageSize: 2 });
          seen.push(jobLogs.loadProgress);
          return jobLogs;
        },
        { wrapper: createWrapper(), initialProps: { name: JOB_NAME } }
      );

      await act(async () => {
        await tick();
      });
      expect(result.current.loadProgress).toEqual({ loaded: 2, total: 2 });

      const nextJob = deferred<HelixJobLogPage>();
      mockJobsPageJobLogs.mockReturnValueOnce(nextJob.promise);

      seen.length = 0;
      rerender({ name: 'other-job' });

      expect(seen.length).toBeGreaterThan(0);
      expect(seen).toEqual(seen.map(() => null));

      await act(async () => {
        nextJob.resolve(makePage([makeLog(0)], 1));
        await tick();
      });
      expect(result.current.loadProgress).toEqual({ loaded: 1, total: 1 });
    });

    it('reports the same progress to every observer of the job', async () => {
      mockJobsPageJobLogs
        .mockResolvedValueOnce(makePage([makeLog(0), makeLog(1)], 4, 'c1'))
        .mockResolvedValueOnce(makePage([makeLog(2), makeLog(3)], 4));

      // A log panel and a download button on the same route observe one shared query
      // key, and only one of them owns the queryFn that walks the pages. Both have to
      // see the walk.
      const { result } = renderHook(
        () => ({
          viewer: useJobLogs({ workspace: WORKSPACE, name: JOB_NAME, pageSize: 2 }),
          downloader: useJobLogs({
            workspace: WORKSPACE,
            name: JOB_NAME,
            pageSize: 2,
            maxPages: Infinity,
          }),
        }),
        { wrapper: createWrapper() }
      );

      await act(async () => {
        await tick();
      });

      expect(result.current.viewer.loadProgress).toEqual({ loaded: 4, total: 4 });
      expect(result.current.downloader.loadProgress).toEqual(result.current.viewer.loadProgress);
    });
  });
});
