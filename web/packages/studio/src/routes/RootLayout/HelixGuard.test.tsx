// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getHelixHealthUrl } from '@studio/api/helixHealth';
import { ROUTES } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { HelixGuard } from '@studio/routes/RootLayout/HelixGuard';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
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
  it('shows a connecting state, then renders children once the platform is ready', async () => {
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

  it('explains when the platform is reachable but still starting', async () => {
    server.use(
      http.get(getHelixHealthUrl(), () =>
        HttpResponse.json({ detail: { status: 'not_ready' } }, { status: 503 })
      )
    );
    renderGuard();

    expect(await screen.findByText('NeMo Helix is still starting')).toBeInTheDocument();
    expect(screen.queryByText('protected content')).not.toBeInTheDocument();
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
});
