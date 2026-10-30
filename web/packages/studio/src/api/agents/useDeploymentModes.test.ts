// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAgentsListDeploymentModes } from '@nemo/sdk/generated/agents/agent-deployments';
import { enabledImageModes, useDeploymentModes } from '@studio/api/agents/useDeploymentModes';
import {
  executionProfilesResponse,
  DEPLOYMENT_MODES_URL as modesUrl,
  EXECUTION_PROFILES_URL as profilesUrl,
} from '@studio/mocks/handlers/agentDeploymentCapabilities';
import { server } from '@studio/mocks/node';
import {
  mockEnabledModes,
  mockExecutionProfiles,
} from '@studio/tests/util/mockAgentDeploymentCapabilities';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { act, renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

describe('useDeploymentModes', () => {
  it('lists the enabled modes once they arrive', async () => {
    mockEnabledModes('subprocess', 'k8s');
    const { result } = renderHook(() => useDeploymentModes('default'), {
      wrapper: TestProviders,
    });

    expect(result.current).toEqual({ status: 'loading' });
    await waitFor(() =>
      expect(result.current).toEqual({
        status: 'ready',
        enabled: ['subprocess', 'k8s'],
        withoutImage: ['subprocess'],
      })
    );
  });

  it('is unknown when the modes cannot be read', async () => {
    server.use(http.get(modesUrl, () => HttpResponse.json({}, { status: 403 })));
    const { result } = renderHook(() => useDeploymentModes('default'), {
      wrapper: TestProviders,
    });

    await waitFor(() => expect(result.current).toEqual({ status: 'unknown' }));
  });

  it('keeps the last good modes when a refetch fails', async () => {
    mockEnabledModes('subprocess', 'k8s');
    const { result } = renderHook(
      () => ({
        modes: useDeploymentModes('default'),
        query: useAgentsListDeploymentModes('default'),
      }),
      { wrapper: TestProviders }
    );
    await waitFor(() => expect(result.current.modes.status).toBe('ready'));

    server.use(http.get(modesUrl, () => HttpResponse.json({}, { status: 403 })));
    await act(async () => {
      await result.current.query.refetch();
    });

    await waitFor(() => expect(result.current.query.isError).toBe(true));
    expect(result.current.modes).toEqual({
      status: 'ready',
      enabled: ['subprocess', 'k8s'],
      withoutImage: ['subprocess'],
    });
  });

  it('leaves subprocess out on a Kubernetes install', async () => {
    mockEnabledModes('subprocess', 'k8s');
    mockExecutionProfiles([{ profile: 'default', backend: 'kubernetes_job' }]);
    const { result } = renderHook(() => useDeploymentModes('default'), {
      wrapper: TestProviders,
    });

    await waitFor(() =>
      expect(result.current).toEqual({ status: 'ready', enabled: ['k8s'], withoutImage: [] })
    );
  });

  it('keeps subprocess when the profiles cannot be read', async () => {
    mockEnabledModes('subprocess', 'k8s');
    server.use(http.get(profilesUrl, () => HttpResponse.json({}, { status: 403 })));
    const { result } = renderHook(() => useDeploymentModes('default'), {
      wrapper: TestProviders,
    });

    await waitFor(() =>
      expect(result.current).toEqual({
        status: 'ready',
        enabled: ['subprocess', 'k8s'],
        withoutImage: ['subprocess'],
      })
    );
  });

  it('stays loading until the profiles arrive', async () => {
    mockEnabledModes('subprocess', 'k8s');
    let releaseProfiles = () => {};
    const profilesReleased = new Promise<void>((resolve) => {
      releaseProfiles = resolve;
    });
    server.use(
      http.get(profilesUrl, async () => {
        await profilesReleased;
        return HttpResponse.json(
          executionProfilesResponse([{ profile: 'default', backend: 'kubernetes_job' }])
        );
      })
    );
    const { result } = renderHook(
      () => ({
        modes: useDeploymentModes('default'),
        query: useAgentsListDeploymentModes('default'),
      }),
      { wrapper: TestProviders }
    );

    await waitFor(() => expect(result.current.query.isSuccess).toBe(true));
    expect(result.current.modes).toEqual({ status: 'loading' });

    releaseProfiles();
    await waitFor(() =>
      expect(result.current.modes).toEqual({ status: 'ready', enabled: ['k8s'], withoutImage: [] })
    );
  });
});

describe('enabledImageModes', () => {
  it('keeps the enabled container modes in preference order', () => {
    expect(
      enabledImageModes({
        status: 'ready',
        enabled: ['k8s', 'subprocess', 'docker'],
        withoutImage: [],
      })
    ).toEqual(['docker', 'k8s']);
  });

  it('is empty when only subprocess is enabled', () => {
    expect(
      enabledImageModes({ status: 'ready', enabled: ['subprocess'], withoutImage: [] })
    ).toEqual([]);
  });

  it('assumes every container mode until the modes are known', () => {
    expect(enabledImageModes({ status: 'loading' })).toEqual(['docker', 'k8s']);
    expect(enabledImageModes({ status: 'unknown' })).toEqual(['docker', 'k8s']);
  });
});
