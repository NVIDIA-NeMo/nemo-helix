// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { resetGuardrailMocks } from '@studio/mocks/handlers/guardrails';
import { server } from '@studio/mocks/node';
import { GuardrailsRoute } from '@studio/routes/guardrails/GuardrailsRoute';
import { getGuardrailDetailRoute, getGuardrailsRoute } from '@studio/routes/utils';
import { XL_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { useLocation } from 'react-router';

const WORKSPACE = 'default';

const LocationProbe = () => {
  const location = useLocation();
  return <div data-testid="detail-location">{location.pathname}</div>;
};

const renderList = () =>
  renderRoute(undefined, {
    history: getGuardrailsRoute(WORKSPACE),
    routes: [
      {
        path: ROUTES.workspace.guardrails,
        element: <GuardrailsRoute />,
      },
      {
        path: ROUTES.workspace.guardrailDetail,
        element: <LocationProbe />,
      },
    ],
  });

const rowCheckboxAt = (index: number) =>
  screen.getAllByRole('checkbox', { name: /(De)?select row/i })[index];

const selectRow = async (user: ReturnType<typeof userEvent.setup>, index: number) => {
  await waitFor(() => expect(rowCheckboxAt(index)).toBeEnabled());
  if (!(rowCheckboxAt(index) as HTMLInputElement).checked) {
    await user.click(rowCheckboxAt(index));
  }
  await waitFor(() => expect(rowCheckboxAt(index)).toBeChecked());
};

describe('GuardrailsRoute', () => {
  beforeEach(() => {
    resetGuardrailMocks();
  });

  it('navigates to the detail route when a row is clicked', async () => {
    const user = userEvent.setup();
    renderList();

    const row = await screen.findByText('pii-filter', undefined, { timeout: XL_SELECTOR_TIMEOUT });
    await user.click(row);

    expect(await screen.findByTestId('detail-location')).toHaveTextContent(
      getGuardrailDetailRoute(WORKSPACE, 'pii-filter')
    );
  });

  describe('bulk delete', () => {
    it('shows a confirmation modal with the selected count when bulk Delete is clicked', async () => {
      const user = userEvent.setup();
      renderList();

      await screen.findByText('pii-filter', undefined, { timeout: XL_SELECTOR_TIMEOUT });
      await selectRow(user, 0);
      await selectRow(user, 1);

      await user.click(screen.getByRole('button', { name: 'Delete selected guardrails' }));

      const dialog = await screen.findByRole('dialog');
      expect(
        within(dialog).getByRole('heading', { name: 'Delete 2 guardrail configs?' })
      ).toBeInTheDocument();
    });

    it('calls DELETE for each selected config and closes the modal on confirm', async () => {
      const deletedNames: string[] = [];
      server.use(
        http.delete(
          `${PLATFORM_BASE_URL}/apis/guardrails/v2/workspaces/:workspace/configs/:name`,
          ({ params }) => {
            deletedNames.push(String(params.name));
            return new HttpResponse(null, { status: 200 });
          }
        )
      );

      const user = userEvent.setup();
      renderList();

      await screen.findByText('pii-filter', undefined, { timeout: XL_SELECTOR_TIMEOUT });
      await selectRow(user, 0);
      await selectRow(user, 1);

      await user.click(screen.getByRole('button', { name: 'Delete selected guardrails' }));
      const dialog = await screen.findByRole('dialog');
      await user.click(within(dialog).getByRole('button', { name: 'Delete 2 Guardrail Configs' }));

      await waitFor(() =>
        expect([...deletedNames].sort()).toEqual(['pii-filter', 'toxicity-guard'])
      );
      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument(), {
        timeout: XL_SELECTOR_TIMEOUT,
      });
    });

    it('shows a singular title when only one config is selected', async () => {
      const user = userEvent.setup();
      renderList();

      await screen.findByText('pii-filter', undefined, { timeout: XL_SELECTOR_TIMEOUT });
      await selectRow(user, 0);

      await user.click(screen.getByRole('button', { name: 'Delete selected guardrails' }));

      const dialog = await screen.findByRole('dialog');
      expect(
        within(dialog).getByRole('heading', { name: "Delete guardrail config 'pii-filter'?" })
      ).toBeInTheDocument();
    });
  });
});
