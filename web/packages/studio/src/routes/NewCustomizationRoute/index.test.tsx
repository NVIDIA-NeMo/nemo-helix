// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { NewCustomizationFormRoute } from '@studio/routes/NewCustomizationFormRoute';
import { NewCustomizationRoute } from '@studio/routes/NewCustomizationRoute';
import { LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
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

  it('navigates to /new/scratch when "Build from scratch" is clicked', async () => {
    const user = userEvent.setup();
    renderRoute();

    await user.click(screen.getByRole('radio', { name: /Build from scratch/ }));

    expect((await screen.findByTestId(LOCATION_DISPLAY_TEST_ID)).textContent).toBe(
      '/workspaces/default/fine-tune/new/scratch'
    );
  });

  it('deletes a saved template from the form and comes back to the cards without it', async () => {
    const TEMPLATES_URL = `${PLATFORM_BASE_URL}/apis/customization/v2/workspaces/:workspace/job-templates`;
    let deleted = false;
    server.use(
      http.get(TEMPLATES_URL, () =>
        deleted
          ? // Never settles: the card has to be gone from the cache, not from a refetch.
            new Promise<never>(() => {})
          : HttpResponse.json({
              data: [
                {
                  id: 'tpl-1',
                  name: 'my-sft-recipe',
                  workspace: 'default',
                  backend: 'automodel',
                  config: {
                    model: 'default/base-model',
                    dataset: { training: 'default/data' },
                    training: { training_type: 'sft', finetuning_type: 'lora' },
                  },
                },
              ],
              object: 'list',
            })
      ),
      http.delete(`${TEMPLATES_URL}/:name`, () => {
        deleted = true;
        return new HttpResponse(null, { status: 204 });
      })
    );
    const user = userEvent.setup();
    const router = createMemoryRouter(
      [
        { path: ROUTES.workspace.newCustomizationJob, element: <NewCustomizationRoute /> },
        { path: ROUTES.workspace.newCustomizationJobScratch, element: <NewCustomizationFormRoute /> },
      ],
      { initialEntries: ['/workspaces/default/fine-tune/new'] }
    );
    render(
      <TestProviders>
        <RouterProvider router={router} />
      </TestProviders>
    );

    await user.click(await screen.findByRole('radio', { name: /my-sft-recipe/ }));
    await user.click(await screen.findByRole('button', { name: /Delete template/ }));
    await user.click(await screen.findByRole('button', { name: 'Delete' }));

    expect(
      await screen.findByRole('radiogroup', { name: 'How do you want to start?' })
    ).toBeInTheDocument();
    expect(screen.queryByRole('radio', { name: /my-sft-recipe/ })).not.toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /Build from scratch/ })).toBeInTheDocument();
  });
});
