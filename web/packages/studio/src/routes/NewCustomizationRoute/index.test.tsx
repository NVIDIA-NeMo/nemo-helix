// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { NewCustomizationRoute } from '@studio/routes/NewCustomizationRoute';
import { LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, RouterProvider } from 'react-router';

const renderRoute = () => {
  const router = createMemoryRouter(
    [
      { path: ROUTES.workspace.newCustomizationJob, element: <NewCustomizationRoute /> },
      { path: ROUTES.workspace.newCustomizationJobScratch, element: <LocationDisplay /> },
    ],
    { initialEntries: ['/workspaces/default/fine-tune/new'] }
  );
  return render(
    <TestProviders>
      <RouterProvider router={router} />
    </TestProviders>
  );
};

describe('NewCustomizationRoute', () => {
  it('shows the start picker rather than the form', () => {
    renderRoute();
    expect(
      screen.getByRole('radiogroup', { name: 'How do you want to start?' })
    ).toBeInTheDocument();
  });

  it('navigates to /new/scratch when "Build from scratch" is confirmed', async () => {
    const user = userEvent.setup();
    renderRoute();

    await user.click(screen.getByText('Build from scratch'));
    await user.click(screen.getByRole('button', { name: /continue/i }));

    expect((await screen.findByTestId(LOCATION_DISPLAY_TEST_ID)).textContent).toBe(
      '/workspaces/default/fine-tune/new/scratch'
    );
  });
});
