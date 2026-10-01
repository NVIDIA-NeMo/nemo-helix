// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { enabledImageModes, useDeploymentModes } from '@studio/api/agents/useDeploymentModes';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

const modesUrl = `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/deployment-modes`;

const mockEnabledModes = (...enabled: string[]) =>
  server.use(
    http.get(modesUrl, () =>
      HttpResponse.json({
        data: ['subprocess', 'docker', 'k8s'].map((mode) => ({
          mode,
          enabled: enabled.includes(mode),
          requires_image: mode !== 'subprocess',
        })),
      })
    )
  );

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
