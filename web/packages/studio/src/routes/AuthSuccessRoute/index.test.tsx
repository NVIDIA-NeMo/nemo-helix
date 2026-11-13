// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { createToastMock } from '@nemo/common/src/providers/toast/useToastMock';
import { logger } from '@nemo/common/src/utils/logger';
import {
  entitiesCreateWorkspace,
  entitiesGetWorkspace,
  getEntitiesGetWorkspaceQueryKey,
} from '@nemo/sdk/generated/platform/entity-store';
import type { AuthProfile } from '@studio/providers/auth';
import { AuthSuccessRoute } from '@studio/routes/AuthSuccessRoute';
import { getWorkspaceDetailsDefaultRoute } from '@studio/routes/utils';
import { mockUseNavigate } from '@studio/tests/util/mockUseParams';
import { render } from '@studio/tests/util/render';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { QueryClient } from '@tanstack/react-query';
import { waitFor } from '@testing-library/react';
import { AxiosError, AxiosHeaders } from 'axios';

vi.mock('@nemo/common/src/providers/toast/useToast', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/common/src/providers/toast/useToast')>()),
  useToast: vi.fn(),
}));

vi.mock('@nemo/common/src/utils/logger', () => ({
  logger: { error: vi.fn() },
}));

vi.mock('@nemo/sdk/generated/platform/entity-store', () => ({
  entitiesCreateWorkspace: vi.fn(),
  entitiesGetWorkspace: vi.fn(),
  getEntitiesGetWorkspaceQueryKey: (name: string) => [`/apis/entities/v2/workspaces/${name}`],
}));

let mockProfile: AuthProfile;
let toast = createToastMock();

vi.mock('@studio/providers/auth', () => ({
  useAuthProfile: () => mockProfile,
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
    mockProfile = { workspace: 'alray', name: 'alray@nvidia.com', email: 'alray@nvidia.com' };
    toast = createToastMock();
    vi.mocked(useToast).mockReturnValue(toast);
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
    mockGetWorkspace
      .mockRejectedValueOnce(axiosErrorWithStatus(404))
      .mockResolvedValueOnce({ name: 'alray' } as never);
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

  describe('when the personal workspace cannot be opened', () => {
    const defaultRoute = getWorkspaceDetailsDefaultRoute(DEFAULT_WORKSPACE);
    const navigate = vi.fn();

    beforeEach(() => {
      mockUseNavigate(navigate);
    });

    const renderRoute = () => {
      render(
        <TestProviders>
          <AuthSuccessRoute />
        </TestProviders>
      );
    };

    it('lands on the default workspace when workspace creation is forbidden', async () => {
      mockGetWorkspace.mockRejectedValue(axiosErrorWithStatus(403));
      mockCreateWorkspace.mockRejectedValue(axiosErrorWithStatus(403));

      renderRoute();

      await waitFor(() => {
        expect(navigate).toHaveBeenCalledWith(defaultRoute);
      });
      expect(toast.error).not.toHaveBeenCalled();
      expect(logger.error).not.toHaveBeenCalled();
    });

    it('lands on the default workspace when the workspace exists but is not readable', async () => {
      mockGetWorkspace.mockRejectedValue(axiosErrorWithStatus(403));
      mockCreateWorkspace.mockRejectedValue(axiosErrorWithStatus(409));

      renderRoute();

      await waitFor(() => {
        expect(navigate).toHaveBeenCalledWith(defaultRoute);
      });
      expect(mockGetWorkspace).toHaveBeenCalledTimes(2);
      expect(toast.error).not.toHaveBeenCalled();
    });

    it.each([
      {
        scenario: 'the lookup fails',
        setup: () => mockGetWorkspace.mockRejectedValue(axiosErrorWithStatus(500)),
      },
      {
        scenario: 'the create fails',
        setup: () => {
          mockGetWorkspace.mockRejectedValue(axiosErrorWithStatus(404));
          mockCreateWorkspace.mockRejectedValue(axiosErrorWithStatus(500));
        },
      },
      {
        scenario: 'the access check after a create conflict fails',
        setup: () => {
          mockGetWorkspace
            .mockRejectedValueOnce(axiosErrorWithStatus(403))
            .mockRejectedValueOnce(axiosErrorWithStatus(500));
          mockCreateWorkspace.mockRejectedValue(axiosErrorWithStatus(409));
        },
      },
    ])('reports an unexpected failure when $scenario', async ({ setup }) => {
      setup();

      renderRoute();

      await waitFor(() => {
        expect(navigate).toHaveBeenCalledWith(defaultRoute);
      });
      expect(toast.error).toHaveBeenCalledExactlyOnceWith(
        "Couldn't set up your personal workspace alray."
      );
      expect(logger.error).toHaveBeenCalled();
    });

    it('ignores a return path into the personal workspace', async () => {
      mockProfile.state = { path: '/workspaces/alray/agents', search: '' };
      mockGetWorkspace.mockRejectedValue(axiosErrorWithStatus(403));
      mockCreateWorkspace.mockRejectedValue(axiosErrorWithStatus(403));

      renderRoute();

      await waitFor(() => {
        expect(navigate).toHaveBeenCalledWith(defaultRoute);
      });
    });

    it.each(['/workspaces', '/workspaces/', '/workspaces//'])(
      'ignores the %s return path, which redirects to the personal workspace',
      async (path) => {
        mockProfile.state = { path, search: '' };
        mockGetWorkspace.mockRejectedValue(axiosErrorWithStatus(403));
        mockCreateWorkspace.mockRejectedValue(axiosErrorWithStatus(403));

        renderRoute();

        await waitFor(() => {
          expect(navigate).toHaveBeenCalledWith(defaultRoute);
        });
      }
    );

    it('keeps a return path into another workspace', async () => {
      mockProfile.state = { path: '/workspaces/team/agents', search: '?tab=1' };
      mockGetWorkspace.mockRejectedValue(axiosErrorWithStatus(403));
      mockCreateWorkspace.mockRejectedValue(axiosErrorWithStatus(403));

      renderRoute();

      await waitFor(() => {
        expect(navigate).toHaveBeenCalledWith({
          pathname: '/workspaces/team/agents',
          search: '?tab=1',
        });
      });
    });
  });
});
