// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { useJobNotifications } from '@studio/hooks/useJobNotifications';
import { server } from '@studio/mocks/node';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import type { FC, PropsWithChildren } from 'react';
import { MemoryRouter } from 'react-router';

const JOBS_URL = `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs`;

const jobsResponse = (status: string, createdAt = '2026-10-13T00:00:00Z') => ({
  data: [
    {
      id: 'j1',
      name: 'my-job',
      workspace: DEFAULT_WORKSPACE,
      source: 'customization',
      status,
      created_at: createdAt,
    },
  ],
  pagination: { page: 1, page_size: 50, total_results: 1, total_pages: 1 },
});

const notificationCtor = vi.fn();
const requestPermission = vi.fn().mockResolvedValue('granted');

class FakeNotification {
  static permission = 'granted';
  static requestPermission = requestPermission;
  onclick: (() => void) | null = null;
  close = vi.fn();
  constructor(title: string, options: NotificationOptions) {
    notificationCtor(title, options);
  }
}

const setup = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const wrapper: FC<PropsWithChildren> = ({ children }) => (
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>
  );
  renderHook(() => useJobNotifications(DEFAULT_WORKSPACE), { wrapper });
  return queryClient;
};

const serveStatus = (status: string, createdAt?: string) =>
  server.use(http.get(JOBS_URL, () => HttpResponse.json(jobsResponse(status, createdAt))));

describe('useJobNotifications', () => {
  beforeEach(() => {
    notificationCtor.mockClear();
    requestPermission.mockClear();
    FakeNotification.permission = 'granted';
    vi.stubGlobal('Notification', FakeNotification);
    vi.spyOn(document, 'hasFocus').mockReturnValue(false);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('notifies when a running job completes while the user is away', async () => {
    serveStatus('active');
    const queryClient = setup();
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    expect(notificationCtor).not.toHaveBeenCalled();

    serveStatus('completed');
    await act(() => queryClient.invalidateQueries());

    await waitFor(() => expect(notificationCtor).toHaveBeenCalledTimes(1));
    expect(notificationCtor).toHaveBeenCalledWith(
      'Job completed',
      expect.objectContaining({ tag: 'my-job', silent: false })
    );
  });

  it('plays the failure chime alongside the notification', async () => {
    const oscillatorFrequencies: number[] = [];
    class FakeAudioContext {
      currentTime = 0;
      destination = {};
      resume = vi.fn().mockResolvedValue(undefined);
      createGain = () => ({
        gain: {
          setValueAtTime: vi.fn(),
          linearRampToValueAtTime: vi.fn(),
          exponentialRampToValueAtTime: vi.fn(),
        },
        connect: vi.fn(),
      });
      createOscillator = () => ({
        frequency: {
          set value(hz: number) {
            oscillatorFrequencies.push(hz);
          },
        },
        connect: vi.fn(),
        start: vi.fn(),
        stop: vi.fn(),
      });
    }
    vi.stubGlobal('AudioContext', FakeAudioContext);

    serveStatus('active');
    const queryClient = setup();
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));

    serveStatus('error');
    await act(() => queryClient.invalidateQueries());

    await waitFor(() => expect(notificationCtor).toHaveBeenCalledTimes(1));
    // Failure is the lower, falling pair.
    expect(oscillatorFrequencies).toHaveLength(2);
    expect(oscillatorFrequencies[1]).toBeLessThan(oscillatorFrequencies[0]);
  });

  it('notifies when a running job fails', async () => {
    serveStatus('active');
    const queryClient = setup();
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));

    serveStatus('error');
    await act(() => queryClient.invalidateQueries());

    await waitFor(() =>
      expect(notificationCtor).toHaveBeenCalledWith('Job failed', expect.anything())
    );
  });

  it('notifies, once, for a job started this session that fails before it is ever seen running', async () => {
    serveStatus('error', new Date().toISOString());
    const queryClient = setup();

    await waitFor(() =>
      expect(notificationCtor).toHaveBeenCalledWith('Job failed', expect.anything())
    );

    // The finished job stays in the list; later polls must not announce it again.
    await act(() => queryClient.invalidateQueries());
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    expect(notificationCtor).toHaveBeenCalledTimes(1);
  });

  it('does not notify for a job first seen already finished', async () => {
    serveStatus('completed');
    const queryClient = setup();
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    expect(notificationCtor).not.toHaveBeenCalled();
  });

  it('stays quiet while the user is looking at Studio', async () => {
    vi.spyOn(document, 'hasFocus').mockReturnValue(true);
    vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible');
    serveStatus('active');
    const queryClient = setup();
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));

    serveStatus('completed');
    await act(() => queryClient.invalidateQueries());
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));

    expect(notificationCtor).not.toHaveBeenCalled();
  });

  it('keeps polling while the tab is in the background', async () => {
    // TanStack Query pauses interval refetching in a background tab by default, which would
    // silence the watcher exactly when the user has switched away.
    serveStatus('active');
    const queryClient = setup();
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));

    const [query] = queryClient.getQueryCache().getAll();
    expect(query.observers[0].options.refetchIntervalInBackground).toBe(true);
  });

  it('asks for permission once a running job is seen and permission is undecided', async () => {
    FakeNotification.permission = 'default';
    serveStatus('active');
    setup();
    await waitFor(() => expect(requestPermission).toHaveBeenCalledTimes(1));
  });

  it('does not ask again when permission was already decided', async () => {
    FakeNotification.permission = 'denied';
    serveStatus('active');
    const queryClient = setup();
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    expect(requestPermission).not.toHaveBeenCalled();
  });
});
