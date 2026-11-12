// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { TaskListRoute } from '@studio/routes/evaluation/tasks/TaskListRoute';
import { renderRoute } from '@studio/tests/util/render';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const renderList = (workspace = 'default') =>
  renderRoute(undefined, {
    history: `/workspaces/${workspace}/evaluation/tasks`,
    routes: [
      { path: ROUTES.workspace.tasks, element: <TaskListRoute /> },
      { path: ROUTES.workspace.taskDetail, element: <div>detail page</div> },
    ],
  });

describe('TaskListRoute', () => {
  it('lists the workspace tasks with runner, instruction, and latest revision', async () => {
    renderList();
    const ints = await screen.findByRole('row', { name: /\bsort-ints\b/ });
    expect(within(ints).getByText('Harbor')).toBeInTheDocument();
    expect(within(ints).getByText(/Sort the integers in input.txt/)).toBeInTheDocument();
    expect(within(ints).getByText('r1')).toBeInTheDocument();
    const strings = screen.getByRole('row', { name: /\bsort-strings\b/ });
    expect(within(strings).getByText('Inline')).toBeInTheDocument();
    expect(
      within(strings).getByText('Sort a list of strings case-insensitively')
    ).toBeInTheDocument();
    expect(within(strings).getByText('r2')).toBeInTheDocument();
  });

  it('opens a task on row click', async () => {
    renderList();
    await userEvent.click(await screen.findByText('sort-ints'));
    expect(await screen.findByText('detail page')).toBeInTheDocument();
  });

  it('is read-only and shows the first-use empty state in a workspace with no tasks', async () => {
    renderList('empty');
    expect(await screen.findByText('No tasks yet')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /create/i })).not.toBeInTheDocument();
  });
});
