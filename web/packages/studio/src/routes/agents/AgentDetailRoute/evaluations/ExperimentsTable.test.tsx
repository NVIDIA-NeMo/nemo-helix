// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { getGetExperimentQueryKey } from '@nemo/sdk/generated/platform/experiments';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { ExperimentsTable } from '@studio/routes/agents/AgentDetailRoute/evaluations/ExperimentsTable';
import { render, screen } from '@studio/tests/util/render';
import { waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

it('warns that deleting selected experiments also deletes their sole-membership evaluations', async () => {
  const deleted = vi.fn(() => new HttpResponse(null, { status: 204 }));
  server.use(http.delete(mockApiUrl(getGetExperimentQueryKey, ':workspace', ':name'), deleted));
  render(
    <ExperimentsTable
      workspace={DEFAULT_WORKSPACE}
      experiments={[
        {
          id: 'experiment',
          name: 'sample',
          evaluationCount: 3,
          runCount: 5,
          latestCreatedAt: null,
        },
      ]}
    />
  );
  const selectAll = (await screen.findAllByRole('checkbox'))[0];
  await userEvent.click(selectAll);
  await userEvent.click(await screen.findByRole('button', { name: 'Delete selected experiments' }));
  const dialog = await screen.findByRole('dialog');
  expect(within(dialog).getByText('Delete 1 experiment?')).toBeVisible();
  expect(
    within(dialog).getByText(
      /Evaluations belonging only to the selected experiments will also be deleted/
    )
  ).toBeVisible();
  expect(deleted).not.toHaveBeenCalled();
  await userEvent.click(within(dialog).getByRole('button', { name: 'Delete' }));
  await waitFor(() => expect(deleted).toHaveBeenCalledOnce());
});
