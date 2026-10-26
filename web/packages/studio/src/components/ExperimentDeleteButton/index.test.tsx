// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { getGetExperimentQueryKey } from '@nemo/sdk/generated/platform/experiments';
import { ExperimentDeleteButton } from '@studio/components/ExperimentDeleteButton';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { render, screen } from '@studio/tests/util/render';
import { waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const NAME = 'sample-experiment';

it.each([false, true])(
  'requires confirmation without card navigation (menu: %s)',
  async (asMenu) => {
    const deleted = vi.fn();
    const navigate = vi.fn();
    const request = vi.fn(() => new HttpResponse(null, { status: 204 }));
    server.use(http.delete(mockApiUrl(getGetExperimentQueryKey, ':workspace', ':name'), request));
    render(
      <div onClick={navigate}>
        <ExperimentDeleteButton
          workspace={DEFAULT_WORKSPACE}
          name={NAME}
          onDeleted={deleted}
          asMenu={asMenu}
        />
      </div>
    );
    const openDelete = async () => {
      await userEvent.click(
        screen.getByRole('button', {
          name: asMenu ? `More actions for experiment ${NAME}` : `Delete experiment ${NAME}`,
        })
      );
      if (asMenu) await userEvent.click(await screen.findByRole('menuitem', { name: 'Delete' }));
    };
    await openDelete();
    const dialog = await screen.findByRole('dialog');
    expect(
      within(dialog).getByText(
        /Evaluations belonging only to the selected experiments will also be deleted/
      )
    ).toBeVisible();
    expect(request).not.toHaveBeenCalled();
    await userEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
    expect(request).not.toHaveBeenCalled();
    await openDelete();
    await userEvent.click(
      within(await screen.findByRole('dialog')).getByRole('button', { name: 'Delete' })
    );
    await waitFor(() => expect(deleted).toHaveBeenCalledOnce());
    expect(request).toHaveBeenCalledOnce();
    expect(navigate).not.toHaveBeenCalled();
  }
);

it('surfaces deletion errors without navigating away', async () => {
  const deleted = vi.fn();
  server.use(
    http.delete(mockApiUrl(getGetExperimentQueryKey, ':workspace', ':name'), () =>
      HttpResponse.json({ detail: 'Experiment deletion failed' }, { status: 500 })
    )
  );
  render(<ExperimentDeleteButton workspace={DEFAULT_WORKSPACE} name={NAME} onDeleted={deleted} />);
  await userEvent.click(screen.getByRole('button', { name: `Delete experiment ${NAME}` }));
  await userEvent.click(
    within(await screen.findByRole('dialog')).getByRole('button', { name: 'Delete' })
  );
  expect(await screen.findByText(/Experiment deletion failed/)).toBeVisible();
  expect(deleted).not.toHaveBeenCalled();
});
