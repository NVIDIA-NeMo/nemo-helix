// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getHelixHealthUrl, HELIX_HEALTH_RETRY_INTERVAL_MS } from '@studio/api/helixHealth';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { HelixGuard } from '@studio/routes/RootLayout/HelixGuard';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { MemoryRouter } from 'react-router';

const renderGuard = (initialPath = '/workspaces/default') =>
  render(
    <TestProviders>
      <MemoryRouter initialEntries={[initialPath]}>
        <HelixGuard>
          <div>protected content</div>
        </HelixGuard>
      </MemoryRouter>
    </TestProviders>
  );

const platformDown = () => server.use(http.get(getHelixHealthUrl(), () => HttpResponse.error()));

describe('HelixGuard', () => {
  it('shows a connecting state, then renders children once the platform is live', async () => {
    renderGuard();
    expect(screen.getByText('Connecting to NeMo Helix...')).toBeInTheDocument();
    expect(await screen.findByText('protected content')).toBeInTheDocument();
  });

  it('blocks the app when the platform is unreachable', async () => {
    platformDown();
    renderGuard();

    expect(await screen.findByText("Studio can't connect to NeMo Helix")).toBeInTheDocument();
    const help = screen.getByTestId('helix-unavailable-help');
    expect(help).toHaveTextContent('nemo services run');
    expect(help).toHaveTextContent(getHelixHealthUrl());
    expect(screen.queryByText('protected content')).not.toBeInTheDocument();
  });

  it('allows access when the platform is live even if readiness fails', async () => {
    const readinessProbe = vi.fn(() =>
      HttpResponse.json({ detail: { status: 'not_ready' } }, { status: 503 })
    );
    server.use(http.get(`${PLATFORM_BASE_URL}/health/ready`, readinessProbe));
    renderGuard();

    expect(await screen.findByText('protected content')).toBeInTheDocument();
    expect(readinessProbe).not.toHaveBeenCalled();
    expect(screen.queryByText("Studio can't connect to NeMo Helix")).not.toBeInTheDocument();
  });

  it('retries on demand and unblocks once the platform comes up', async () => {
    const user = userEvent.setup();
    platformDown();
    renderGuard();
    await screen.findByText("Studio can't connect to NeMo Helix");

    server.resetHandlers();
    await user.click(screen.getByRole('button', { name: 'Retry connection' }));

    expect(await screen.findByText('protected content')).toBeInTheDocument();
  });

  it('never blocks the auth callback route', () => {
    platformDown();
    renderGuard(ROUTES.auth.success);
    expect(screen.getByText('protected content')).toBeInTheDocument();
  });

  it('recovers automatically after a failed probe and stops polling after success', async () => {
    let isLive = false;
    const livenessProbe = vi.fn(() =>
      isLive ? HttpResponse.json({ status: 'live' }) : HttpResponse.error()
    );
    server.use(http.get(getHelixHealthUrl(), livenessProbe));
    const { unmount } = renderGuard();

    try {
      await screen.findByText("Studio can't connect to NeMo Helix");
      expect(livenessProbe).toHaveBeenCalledTimes(1);

      isLive = true;
      expect(
        await screen.findByText(
          'protected content',
          {},
          {
            timeout: HELIX_HEALTH_RETRY_INTERVAL_MS + 2_000,
          }
        )
      ).toBeInTheDocument();
      expect(livenessProbe).toHaveBeenCalledTimes(2);

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, HELIX_HEALTH_RETRY_INTERVAL_MS + 100));
      });
      expect(livenessProbe).toHaveBeenCalledTimes(2);
    } finally {
      unmount();
    }
  }, 20_000);
});
