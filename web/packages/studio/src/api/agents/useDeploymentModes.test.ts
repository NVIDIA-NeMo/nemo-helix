// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAgentsListDeploymentModes } from '@nemo/sdk/generated/agents/agent-deployments';
import { enabledImageModes, useDeploymentModes } from '@studio/api/agents/useDeploymentModes';
import { DEPLOYMENT_MODES_URL as modesUrl } from '@studio/mocks/handlers/agentDeploymentCapabilities';
import { server } from '@studio/mocks/node';
import { mockEnabledModes } from '@studio/tests/util/mockAgentDeploymentCapabilities';
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
      expect(result.current).toEqual({ status: 'ready', enabled: ['subprocess', 'k8s'] })
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
    expect(result.current.modes).toEqual({ status: 'ready', enabled: ['subprocess', 'k8s'] });
  });
});

describe('enabledImageModes', () => {
  it('keeps the enabled container modes in preference order', () => {
    expect(
      enabledImageModes({ status: 'ready', enabled: ['k8s', 'subprocess', 'docker'] })
    ).toEqual(['docker', 'k8s']);
  });

  it('is empty when only subprocess is enabled', () => {
    expect(enabledImageModes({ status: 'ready', enabled: ['subprocess'] })).toEqual([]);
  });

  it('assumes every container mode until the modes are known', () => {
    expect(enabledImageModes({ status: 'loading' })).toEqual(['docker', 'k8s']);
    expect(enabledImageModes({ status: 'unknown' })).toEqual(['docker', 'k8s']);
  });
});
