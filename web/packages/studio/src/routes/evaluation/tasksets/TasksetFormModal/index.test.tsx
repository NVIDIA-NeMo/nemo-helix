// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TasksetInput } from '@nemo/sdk/generated/evals/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { TasksetFormModal } from '@studio/routes/evaluation/tasksets/TasksetFormModal';
import { renderRoute } from '@studio/tests/util/render';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const TASKSET_URL = `${PLATFORM_BASE_URL}/apis/evals/v2/workspaces/:workspace/tasksets/:name`;

const renderModal = (onClose = vi.fn()) => {
  renderRoute(undefined, {
    history: '/workspaces/default/evaluation/tasksets',
    routes: [
      {
        path: ROUTES.workspace.tasksets,
        element: <TasksetFormModal open onClose={onClose} />,
      },
      { path: ROUTES.workspace.tasksetDetail, element: <div>detail page</div> },
    ],
  });
  return { onClose };
};

const selectTask = async (name: string) => {
  await userEvent.click(screen.getByRole('combobox', { name: 'Tasks' }));
  await userEvent.click(await screen.findByRole('option', { name }));
  await userEvent.keyboard('{Escape}');
};

describe('TasksetFormModal', () => {
  it('creates the taskset from the selected tasks and opens it', async () => {
    let request: { name: unknown; body: TasksetInput } | undefined;
    server.use(
      http.post(TASKSET_URL, async ({ request: req, params }) => {
        request = { name: params.name, body: (await req.json()) as TasksetInput };
        return HttpResponse.json({}, { status: 201 });
      })
    );
    const { onClose } = renderModal();

    await userEvent.type(screen.getByRole('textbox', { name: 'Name' }), 'My Suite');
    await userEvent.type(
      screen.getByRole('textbox', { name: 'Description (optional)' }),
      'Sorting smoke tests'
    );
    await selectTask('sort-ints');
    await selectTask('sort-strings');
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    expect(await screen.findByText('detail page')).toBeInTheDocument();
    expect(request).toEqual({
      name: 'my-suite',
      body: { tasks: ['sort-ints', 'sort-strings'], description: 'Sorting smoke tests' },
    });
    expect(onClose).toHaveBeenCalled();
  });

  it('requires a name and at least one task', async () => {
    renderModal();
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));
    expect(await screen.findByText('Select at least one task.')).toBeInTheDocument();
    expect(
      screen.getByText('Name must start with a letter and be at least 2 characters.')
    ).toBeInTheDocument();
  });

  it('keeps the modal open and shows the server error when the name is taken', async () => {
    const { onClose } = renderModal();
    await userEvent.type(screen.getByRole('textbox', { name: 'Name' }), 'sort-suite');
    await selectTask('sort-ints');
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    expect(await screen.findByText(/already exists/)).toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();
  });
});
