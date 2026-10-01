// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCanBuildAgentImages } from '@studio/api/agents/useCanBuildAgentImages';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

const profilesUrl = `${PLATFORM_BASE_URL}/apis/jobs/v2/execution-profiles`;

const mockProfiles = (profiles: { profile: string; backend: string }[]) =>
  server.use(
    http.get(profilesUrl, () =>
      HttpResponse.json(profiles.map((profile) => ({ provider: 'cpu', ...profile })))
    )
  );

describe('useCanBuildAgentImages', () => {
  it('is loading until the profiles arrive', () => {
    const { result } = renderHook(() => useCanBuildAgentImages(), { wrapper: TestProviders });

    expect(result.current).toBe('loading');
  });

  it('is supported with a default subprocess profile', async () => {
    mockProfiles([{ profile: 'default', backend: 'subprocess' }]);
    const { result } = renderHook(() => useCanBuildAgentImages(), { wrapper: TestProviders });

    await waitFor(() => expect(result.current).toBe('supported'));
  });

  it('is unsupported when only cluster profiles exist', async () => {
    mockProfiles([{ profile: 'default', backend: 'kubernetes_job' }]);
    const { result } = renderHook(() => useCanBuildAgentImages(), { wrapper: TestProviders });

    await waitFor(() => expect(result.current).toBe('unsupported'));
  });

  it('is unsupported when the subprocess profile has another name', async () => {
    mockProfiles([{ profile: 'builds', backend: 'subprocess' }]);
    const { result } = renderHook(() => useCanBuildAgentImages(), { wrapper: TestProviders });

    await waitFor(() => expect(result.current).toBe('unsupported'));
  });

  it('is unknown when the profiles cannot be read', async () => {
    server.use(http.get(profilesUrl, () => HttpResponse.json({}, { status: 403 })));
    const { result } = renderHook(() => useCanBuildAgentImages(), { wrapper: TestProviders });

    await waitFor(() => expect(result.current).toBe('unknown'));
  });
});
