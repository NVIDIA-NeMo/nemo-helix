// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEntitiesListWorkspaces } from '@nemo/sdk/generated/platform/entity-store';
import type { Workspace } from '@nemo/sdk/generated/platform/schema';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { useHasSampleWorkspace } from '@studio/routes/WorkspaceDashboardHomeRoute/useHasSampleWorkspace';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

const pagedListing = (totalPages: number, ...workspaces: Partial<Workspace>[]) =>
  server.use(
    http.get(`${PLATFORM_BASE_URL}/apis/entities/v2/workspaces`, () =>
      HttpResponse.json({
        object: 'list',
        data: workspaces,
        pagination: {
          page: 1,
          page_size: 1000,
          total_pages: totalPages,
          total_results: workspaces.length,
        },
      })
    )
  );

const listing = (...workspaces: Partial<Workspace>[]) => pagedListing(1, ...workspaces);

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

  it('is true when page 1 has a sample workspace even if more pages exist', async () => {
    pagedListing(2, { name: 'default' }, { name: 'sample-1a2b3c4d' });
    const { result } = renderUseHasSampleWorkspace();

    await waitFor(() => expect(result.current).toBe(true));
  });

  it('stays undefined when page 1 lacks a sample but more pages exist', async () => {
    pagedListing(2, { name: 'default' });
    // Observe the shared query so we can wait for the response before asserting.
    const { result } = renderHook(
      () => ({
        hasSample: useHasSampleWorkspace(),
        query: useEntitiesListWorkspaces({ page: 1, page_size: DEFAULT_LARGE_PAGE_SIZE }),
      }),
      { wrapper: TestProviders }
    );

    await waitFor(() => expect(result.current.query.isSuccess).toBe(true));
    expect(result.current.hasSample).toBeUndefined();
  });

  it('stays undefined while disabled', () => {
    listing({ name: 'sample-1a2b3c4d' });
    const { result } = renderUseHasSampleWorkspace(false);

    expect(result.current).toBeUndefined();
  });
});
