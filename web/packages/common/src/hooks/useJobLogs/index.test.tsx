// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LOGS_MAX_PAGE_SIZE, LOGS_TAIL_SIZE } from '@nemo/common/src/constants';
import { fetchAllJobLogs, useJobLogs } from '@nemo/common/src/hooks/useJobLogs';
import { jobsPageJobLogs } from '@nemo/sdk/generated/platform/jobs';
import type { HelixJobLog, HelixJobLogPage } from '@nemo/sdk/generated/platform/schema';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';

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

function makeLog(index: number, message = `Log message ${index}`): HelixJobLog {
  return {
    timestamp: `2026-01-01T00:00:${String(index).padStart(2, '0')}Z`,
    job: JOB_NAME,
    job_step: 'step',
    job_task: 'task',
    message,
  };
}

const range = (from: number, to: number): HelixJobLog[] =>
  Array.from({ length: to - from }, (_, i) => makeLog(from + i));

/** A page as the tail and cursor modes return it: oldest first, with no `next_page`. */
function makePage(logs: HelixJobLog[], total: number, prevPage = ''): HelixJobLogPage {
  return {
    data: logs,
    total,
    next_page: '',
    prev_page: prevPage,
  };
}

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

async function renderLoaded(firstPage: HelixJobLogPage) {
  mockJobsPageJobLogs.mockResolvedValueOnce(firstPage);
  const view = renderHook(() => useJobLogs({ workspace: WORKSPACE, name: JOB_NAME }), {
    wrapper: createWrapper(),
  });
  await waitFor(() => expect(view.result.current.isLoading).toBe(false));
  return view;
}

describe('useJobLogs', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('loads the newest lines in a single tail request', async () => {
    const { result } = await renderLoaded(makePage(range(0, 3), 3));

    expect(result.current.data).toEqual(range(0, 3));
    expect(result.current.total).toBe(3);
    expect(result.current.error).toBeNull();
    expect(result.current.paging.hasOlderLogs).toBe(false);
    expect(mockJobsPageJobLogs).toHaveBeenCalledTimes(1);
    expect(mockJobsPageJobLogs).toHaveBeenCalledWith(
      WORKSPACE,
      JOB_NAME,
      { tail: LOGS_TAIL_SIZE },
      expect.any(AbortSignal)
    );
  });

  it('reports older lines when the tail comes with a cursor', async () => {
    const { result } = await renderLoaded(makePage(range(5, 8), 8, 'older-1'));

    expect(result.current.data).toEqual(range(5, 8));
    expect(result.current.total).toBe(8);
    expect(result.current.paging.hasOlderLogs).toBe(true);
  });

  it('returns empty array when there are no logs', async () => {
    const { result } = await renderLoaded(makePage([], 0));

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

  describe('polling', () => {
    it('appends the lines a poll brings in', async () => {
      const { result } = await renderLoaded(makePage(range(0, 3), 3));

      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(1, 5), 5, 'older-1'));
      await act(() => result.current.refetch());

      await waitFor(() => expect(result.current.data).toEqual(range(0, 5)));
      expect(result.current.total).toBe(5);
      // Everything from the first load on is held, so nothing older is missing.
      expect(result.current.paging.hasOlderLogs).toBe(false);
      expect(mockJobsPageJobLogs).toHaveBeenCalledTimes(2);
    });

    it('keeps accumulating past the tail window', async () => {
      const { result } = await renderLoaded(makePage(range(0, 3), 3));

      // Four-line windows, three new lines per poll: each window overlaps the last by one.
      for (let end = 6; end <= 12; end += 3) {
        mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(end - 4, end), end, 'older'));
        await act(() => result.current.refetch());
      }

      await waitFor(() => expect(result.current.data).toEqual(range(0, 12)));
    });

    it('takes the server’s view of the overlap, so late lines land in place', async () => {
      const { result } = await renderLoaded(makePage(range(0, 4), 4));

      const late = makeLog(2, 'shipped late');
      mockJobsPageJobLogs.mockResolvedValueOnce(
        makePage([makeLog(1), makeLog(2), late, makeLog(3), makeLog(4)], 6, 'older-1')
      );
      await act(() => result.current.refetch());

      await waitFor(() =>
        expect(result.current.data).toEqual([...range(0, 3), late, ...range(3, 5)])
      );
    });

    it('widens a poll that skips past what is held, keeping loaded older lines', async () => {
      const { result } = await renderLoaded(makePage(range(6, 9), 9, 'older-1'));
      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(0, 6), 9));
      await act(() => result.current.paging.fetchOlderLogs());

      // 14 lines arrived since the last poll, more than the window holds.
      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(20, 23), 23, 'older-poll'));
      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(7, 23), 23, 'older-wide'));
      await act(() => result.current.refetch());

      expect(mockJobsPageJobLogs).toHaveBeenLastCalledWith(
        WORKSPACE,
        JOB_NAME,
        { tail: LOGS_TAIL_SIZE + 14 },
        expect.any(AbortSignal)
      );
      await waitFor(() => expect(result.current.data).toEqual(range(0, 23)));
      expect(result.current.total).toBe(23);
      expect(result.current.paging.hasOlderLogs).toBe(false);
    });

    it('sizes the next poll by how many lines the last one brought in', async () => {
      const { result } = await renderLoaded(makePage(range(0, 3), 3));

      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(2, 17), 17, 'older-1'));
      await act(() => result.current.refetch());
      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(16, 31), 31, 'older-2'));
      await act(() => result.current.refetch());

      expect(mockJobsPageJobLogs).toHaveBeenCalledTimes(3);
      expect(mockJobsPageJobLogs).toHaveBeenLastCalledWith(
        WORKSPACE,
        JOB_NAME,
        { tail: LOGS_TAIL_SIZE + 14 },
        expect.any(AbortSignal)
      );
      await waitFor(() => expect(result.current.data).toEqual(range(0, 31)));
    });

    it('takes the first lines of an empty log in one request', async () => {
      const { result } = await renderLoaded(makePage([], 0));

      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(0, 5), 5));
      await act(() => result.current.refetch());

      await waitFor(() => expect(result.current.data).toEqual(range(0, 5)));
      expect(mockJobsPageJobLogs).toHaveBeenCalledTimes(2);
    });

    it('starts over when even the widened poll does not reach what is held', async () => {
      // More lines arrived than one request can hold.
      const { result } = await renderLoaded(makePage(range(0, 3), 3, 'older-a'));

      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(10, 13), 13, 'older-poll'));
      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(8, 13), 13, 'older-b'));
      await act(() => result.current.refetch());

      await waitFor(() => expect(result.current.data).toEqual(range(8, 13)));

      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(5, 8), 13));
      await act(() => result.current.paging.fetchOlderLogs());

      expect(mockJobsPageJobLogs).toHaveBeenLastCalledWith(WORKSPACE, JOB_NAME, {
        page_cursor: 'older-b',
        limit: LOGS_MAX_PAGE_SIZE,
      });
    });

    it('starts over without widening when a new attempt holds fewer lines', async () => {
      const { result } = await renderLoaded(makePage(range(0, 3), 3));

      const attempt2 = [makeLog(0, 'attempt 2')];
      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(attempt2, 1));
      await act(() => result.current.refetch());

      await waitFor(() => expect(result.current.data).toEqual(attempt2));
      expect(mockJobsPageJobLogs).toHaveBeenCalledTimes(2);
    });

    it("never carries one job's lines into the next", async () => {
      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(0, 3), 3));
      const { result, rerender } = renderHook(
        ({ name }) => useJobLogs({ workspace: WORKSPACE, name }),
        { wrapper: createWrapper(), initialProps: { name: JOB_NAME } }
      );
      await waitFor(() => expect(result.current.data).toEqual(range(0, 3)));

      // Opens with the previous job's last line, which must not stitch the two together.
      const otherJobLogs = [makeLog(2), makeLog(3, 'other job')];
      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(otherJobLogs, 2));
      rerender({ name: 'other-job' });

      await waitFor(() => expect(result.current.data).toEqual(otherJobLogs));
    });
  });

  describe('fetchOlderLogs', () => {
    it('prepends the next page of older lines at the server max', async () => {
      const { result } = await renderLoaded(makePage(range(6, 9), 9, 'older-1'));

      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(3, 6), 9, 'older-2'));
      await act(() => result.current.paging.fetchOlderLogs());

      expect(mockJobsPageJobLogs).toHaveBeenLastCalledWith(WORKSPACE, JOB_NAME, {
        page_cursor: 'older-1',
        limit: LOGS_MAX_PAGE_SIZE,
      });
      expect(result.current.data).toEqual(range(3, 9));
      expect(result.current.paging.hasOlderLogs).toBe(true);

      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(0, 3), 9));
      await act(() => result.current.paging.fetchOlderLogs());

      expect(mockJobsPageJobLogs).toHaveBeenLastCalledWith(WORKSPACE, JOB_NAME, {
        page_cursor: 'older-2',
        limit: LOGS_MAX_PAGE_SIZE,
      });
      expect(result.current.data).toEqual(range(0, 9));
      expect(result.current.paging.hasOlderLogs).toBe(false);
    });

    it('does nothing once the start of the log is held', async () => {
      const { result } = await renderLoaded(makePage(range(0, 3), 3));

      await act(() => result.current.paging.fetchOlderLogs());

      expect(mockJobsPageJobLogs).toHaveBeenCalledTimes(1);
    });

    it('reports the fetch while it is in flight and ignores repeat calls', async () => {
      const { result } = await renderLoaded(makePage(range(6, 9), 9, 'older-1'));

      const older = deferred<HelixJobLogPage>();
      mockJobsPageJobLogs.mockReturnValueOnce(older.promise);
      let firstCall!: Promise<void>;
      act(() => {
        firstCall = result.current.paging.fetchOlderLogs();
      });
      await waitFor(() => expect(result.current.paging.isFetchingOlderLogs).toBe(true));

      await act(() => result.current.paging.fetchOlderLogs());
      expect(mockJobsPageJobLogs).toHaveBeenCalledTimes(2);

      await act(async () => {
        older.resolve(makePage(range(0, 6), 9));
        await firstCall;
      });

      expect(result.current.paging.isFetchingOlderLogs).toBe(false);
      expect(result.current.data).toEqual(range(0, 9));
    });

    it('keeps the older page when a poll lands while it is in flight', async () => {
      const { result } = await renderLoaded(makePage(range(6, 9), 9, 'older-1'));

      const older = deferred<HelixJobLogPage>();
      mockJobsPageJobLogs.mockReturnValueOnce(older.promise);
      let olderCall!: Promise<void>;
      act(() => {
        olderCall = result.current.paging.fetchOlderLogs();
      });

      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(7, 10), 10, 'older-poll'));
      await act(() => result.current.refetch());
      await waitFor(() => expect(result.current.data).toEqual(range(6, 10)));

      await act(async () => {
        older.resolve(makePage(range(0, 6), 10));
        await olderCall;
      });

      expect(result.current.data).toEqual(range(0, 10));
    });

    it('keeps the older page when it lands while a poll is in flight', async () => {
      const { result } = await renderLoaded(makePage(range(6, 9), 9, 'older-1'));

      const poll = deferred<HelixJobLogPage>();
      mockJobsPageJobLogs.mockReturnValueOnce(poll.promise);
      let pollCall!: Promise<unknown>;
      act(() => {
        pollCall = result.current.refetch();
      });

      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(0, 6), 9));
      await act(() => result.current.paging.fetchOlderLogs());
      expect(result.current.data).toEqual(range(0, 9));

      await act(async () => {
        poll.resolve(makePage(range(7, 10), 10, 'older-poll'));
        await pollCall;
      });

      await waitFor(() => expect(result.current.data).toEqual(range(0, 10)));
    });

    it('drops an older page once a poll has started over', async () => {
      const { result } = await renderLoaded(makePage(range(6, 9), 9, 'older-1'));

      const older = deferred<HelixJobLogPage>();
      mockJobsPageJobLogs.mockReturnValueOnce(older.promise);
      let olderCall!: Promise<void>;
      act(() => {
        olderCall = result.current.paging.fetchOlderLogs();
      });

      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(20, 23), 23, 'older-poll'));
      mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(15, 23), 23, 'older-new'));
      await act(() => result.current.refetch());

      await act(async () => {
        older.resolve(makePage(range(0, 6), 23));
        await olderCall;
      });

      expect(result.current.data).toEqual(range(15, 23));
      expect(result.current.paging.hasOlderLogs).toBe(true);
    });

    it('rejects when the request fails, leaving the held lines alone', async () => {
      const { result } = await renderLoaded(makePage(range(6, 9), 9, 'older-1'));

      mockJobsPageJobLogs.mockRejectedValueOnce(new Error('Network failure'));
      await act(() =>
        expect(result.current.paging.fetchOlderLogs()).rejects.toThrow('Network failure')
      );

      expect(result.current.data).toEqual(range(6, 9));
      expect(result.current.paging.isFetchingOlderLogs).toBe(false);
    });
  });
});

describe('fetchAllJobLogs', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('walks back from the tail at the server max and returns the lines oldest first', async () => {
    mockJobsPageJobLogs
      .mockResolvedValueOnce(makePage(range(6, 9), 9, 'older-1'))
      .mockResolvedValueOnce(makePage(range(3, 6), 9, 'older-2'))
      .mockResolvedValueOnce(makePage(range(0, 3), 9));

    await expect(fetchAllJobLogs(WORKSPACE, JOB_NAME)).resolves.toEqual(range(0, 9));

    expect(mockJobsPageJobLogs.mock.calls).toEqual([
      [WORKSPACE, JOB_NAME, { tail: LOGS_MAX_PAGE_SIZE }],
      [WORKSPACE, JOB_NAME, { page_cursor: 'older-1', limit: LOGS_MAX_PAGE_SIZE }],
      [WORKSPACE, JOB_NAME, { page_cursor: 'older-2', limit: LOGS_MAX_PAGE_SIZE }],
    ]);
  });

  it('makes one request when the tail holds the whole log', async () => {
    mockJobsPageJobLogs.mockResolvedValueOnce(makePage(range(0, 3), 3));

    await expect(fetchAllJobLogs(WORKSPACE, JOB_NAME)).resolves.toEqual(range(0, 3));
    expect(mockJobsPageJobLogs).toHaveBeenCalledTimes(1);
  });
});
