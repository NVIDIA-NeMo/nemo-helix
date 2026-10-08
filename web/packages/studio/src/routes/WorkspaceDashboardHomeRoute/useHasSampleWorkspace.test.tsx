// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Workspace } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { useHasSampleWorkspace } from '@studio/routes/WorkspaceDashboardHomeRoute/useHasSampleWorkspace';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

const listing = (...workspaces: Partial<Workspace>[]) =>
  server.use(
    http.get(`${PLATFORM_BASE_URL}/apis/entities/v2/workspaces`, () =>
      HttpResponse.json({
        object: 'list',
        data: workspaces,
        pagination: { page: 1, page_size: 1000, total_pages: 1, total_results: workspaces.length },
      })
    )
  );

const renderUseHasSampleWorkspace = (enabled?: boolean) =>
  renderHook(() => useHasSampleWorkspace(enabled), { wrapper: TestProviders });

describe('useHasSampleWorkspace', () => {
  it('is true when a sample workspace is visible, whoever created it', async () => {
    listing({ name: 'default' }, { name: 'sample-1a2b3c4d', created_by: 'someone-else' });
    const { result } = renderUseHasSampleWorkspace();

    await waitFor(() => expect(result.current).toBe(true));
  });

  it('is false without a sample workspace', async () => {
    listing({ name: 'default' });
    const { result } = renderUseHasSampleWorkspace();

    await waitFor(() => expect(result.current).toBe(false));
  });

  it('is false for a bare "sample" workspace', async () => {
    listing({ name: 'sample' });
    const { result } = renderUseHasSampleWorkspace();

    await waitFor(() => expect(result.current).toBe(false));
  });

  it('stays undefined while disabled', () => {
    listing({ name: 'sample-1a2b3c4d' });
    const { result } = renderUseHasSampleWorkspace(false);

    expect(result.current).toBeUndefined();
  });
});
