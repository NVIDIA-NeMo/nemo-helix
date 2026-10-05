// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useJobsGetExecutionProfiles } from '@nemo/sdk/generated/platform/jobs';
import { useImageBuildsUnsupported } from '@studio/api/agents/useImageBuildsUnsupported';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { mockExecutionProfiles } from '@studio/tests/util/mockAgentDeploymentCapabilities';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

const renderSettled = async () => {
  const { result } = renderHook(
    () => ({ unsupported: useImageBuildsUnsupported(), query: useJobsGetExecutionProfiles() }),
    { wrapper: TestProviders }
  );
  await waitFor(() => expect(result.current.query.isFetched).toBe(true));
  return result.current.unsupported;
};

describe('useImageBuildsUnsupported', () => {
  it('is unsupported when only cluster profiles exist', async () => {
    mockExecutionProfiles([{ profile: 'default', backend: 'kubernetes_job' }]);

    expect(await renderSettled()).toBe(true);
  });

  it('is unsupported when the subprocess profile has another name', async () => {
    mockExecutionProfiles([{ profile: 'builds', backend: 'subprocess' }]);

    expect(await renderSettled()).toBe(true);
  });

  it('is supported with a default subprocess profile', async () => {
    mockExecutionProfiles([{ profile: 'default', backend: 'subprocess' }]);

    expect(await renderSettled()).toBe(false);
  });

  it('does not block builds when the profiles cannot be read', async () => {
    server.use(
      http.get(`${PLATFORM_BASE_URL}/apis/jobs/v2/execution-profiles`, () =>
        HttpResponse.json({}, { status: 403 })
      )
    );

    expect(await renderSettled()).toBe(false);
  });
});
