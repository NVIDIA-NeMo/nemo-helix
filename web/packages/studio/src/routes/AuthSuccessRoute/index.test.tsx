// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  entitiesCreateWorkspace,
  entitiesGetWorkspace,
  getEntitiesGetWorkspaceQueryKey,
} from '@nemo/sdk/generated/platform/entity-store';
import { AuthSuccessRoute } from '@studio/routes/AuthSuccessRoute';
import { mockUseNavigate } from '@studio/tests/util/mockUseParams';
import { render } from '@studio/tests/util/render';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { QueryClient } from '@tanstack/react-query';
import { waitFor } from '@testing-library/react';
import { AxiosError, AxiosHeaders } from 'axios';

vi.mock('@nemo/sdk/generated/platform/entity-store', () => ({
  entitiesCreateWorkspace: vi.fn(),
  entitiesGetWorkspace: vi.fn(),
  getEntitiesGetWorkspaceQueryKey: (name: string) => [`/apis/entities/v2/workspaces/${name}`],
}));

vi.mock('@studio/providers/auth', () => ({
  useAuthProfile: () => ({ workspace: 'alray', name: 'alray@nvidia.com' }),
}));

const mockGetWorkspace = vi.mocked(entitiesGetWorkspace);
const mockCreateWorkspace = vi.mocked(entitiesCreateWorkspace);

const axiosErrorWithStatus = (status: number) =>
  new AxiosError('request failed', undefined, undefined, undefined, {
    status,
    statusText: '',
    data: {},
    headers: {},
    config: { headers: new AxiosHeaders() },
  });

describe('AuthSuccessRoute', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('refreshes the workspace access check before navigating after creating the workspace', async () => {
    mockGetWorkspace.mockRejectedValue(axiosErrorWithStatus(403));
    mockCreateWorkspace.mockResolvedValue({ name: 'alray' } as never);
    const navigate = vi.fn();
    mockUseNavigate(navigate);
    const invalidateQueries = vi.spyOn(QueryClient.prototype, 'invalidateQueries');

    render(
      <TestProviders>
        <AuthSuccessRoute />
      </TestProviders>
    );

    await waitFor(() => {
      expect(navigate).toHaveBeenCalledWith('/');
    });

    expect(mockCreateWorkspace).toHaveBeenCalled();
    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: getEntitiesGetWorkspaceQueryKey('alray'),
    });
    expect(invalidateQueries.mock.invocationCallOrder[0]).toBeLessThan(
      navigate.mock.invocationCallOrder[0]
    );
  });

  it('treats a concurrent create conflict as success', async () => {
    mockGetWorkspace.mockRejectedValue(axiosErrorWithStatus(404));
    mockCreateWorkspace.mockRejectedValue(axiosErrorWithStatus(409));
    const navigate = vi.fn();
    mockUseNavigate(navigate);
    const invalidateQueries = vi.spyOn(QueryClient.prototype, 'invalidateQueries');

    render(
      <TestProviders>
        <AuthSuccessRoute />
      </TestProviders>
    );

    await waitFor(() => {
      expect(navigate).toHaveBeenCalledWith('/');
    });

    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: getEntitiesGetWorkspaceQueryKey('alray'),
    });
  });

  it('still refreshes the access check when the workspace already exists', async () => {
    mockGetWorkspace.mockResolvedValue({ name: 'alray' } as never);
    const navigate = vi.fn();
    mockUseNavigate(navigate);
    const invalidateQueries = vi.spyOn(QueryClient.prototype, 'invalidateQueries');

    render(
      <TestProviders>
        <AuthSuccessRoute />
      </TestProviders>
    );

    await waitFor(() => {
      expect(navigate).toHaveBeenCalledWith('/');
    });

    expect(mockCreateWorkspace).not.toHaveBeenCalled();
    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: getEntitiesGetWorkspaceQueryKey('alray'),
    });
  });
});
