// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { UnauthorizedWorkspace } from '@studio/components/Layouts/UnauthorizedWorkspace';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router';

const renderComponent = () =>
  render(
    <TestProviders>
      <MemoryRouter>
        <UnauthorizedWorkspace />
      </MemoryRouter>
    </TestProviders>
  );

describe('UnauthorizedWorkspace', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders the access denied heading', () => {
    renderComponent();
    expect(screen.getByText("You don't have access to this workspace")).toBeInTheDocument();
  });

  it('renders the permission message', () => {
    renderComponent();
    expect(
      screen.getByText(
        "You don't have permission to view this workspace. Contact the workspace owner to request access."
      )
    ).toBeInTheDocument();
  });

  // The access check can go stale (for example, right after a first-login workspace
  // creation), so the user needs a way out of this screen without hand-reloading.
  it('offers recovery actions', async () => {
    const reload = vi.fn();
    vi.spyOn(window, 'location', 'get').mockReturnValue({
      ...window.location,
      reload,
    } as unknown as Location);

    renderComponent();

    expect(screen.getByRole('button', { name: 'Go Back' })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Refresh Page' }));
    expect(reload).toHaveBeenCalled();
  });
});
