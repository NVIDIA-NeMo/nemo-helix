// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { TasksetListRoute } from '@studio/routes/evaluation/tasksets/TasksetListRoute';
import { renderRoute } from '@studio/tests/util/render';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const renderList = (workspace = 'default') =>
  renderRoute(undefined, {
    history: `/workspaces/${workspace}/evaluation/tasksets`,
    routes: [
      { path: ROUTES.workspace.tasksets, element: <TasksetListRoute /> },
      { path: ROUTES.workspace.tasksetDetail, element: <div>detail page</div> },
    ],
  });

describe('TasksetListRoute', () => {
  it('lists the workspace tasksets with task count, latest revision, and tags', async () => {
    renderList();
    const row = await screen.findByRole('row', { name: /\bsort-suite\b/ });
    expect(within(row).getByText('Sorting algorithm tasks')).toBeInTheDocument();
    expect(within(row).getByText('3')).toBeInTheDocument();
    expect(within(row).getByText('r2')).toBeInTheDocument();
    expect(within(row).getByText('latest')).toBeInTheDocument();
    expect(within(row).getByText('stable')).toBeInTheDocument();
    expect(screen.getByText('phishing-triage')).toBeInTheDocument();
    expect(screen.queryByText('staging-suite')).not.toBeInTheDocument();
  });

  it('opens a taskset on row click', async () => {
    renderList();
    await userEvent.click(await screen.findByText('sort-suite'));
    expect(await screen.findByText('detail page')).toBeInTheDocument();
  });

  it('opens the create modal from the header', async () => {
    renderList();
    await userEvent.click(await screen.findByRole('button', { name: 'Create Taskset' }));
    expect(await screen.findByRole('dialog', { name: 'Create Taskset' })).toBeInTheDocument();
  });

  it('shows the first-use empty state with a create CTA in a workspace with no tasksets', async () => {
    renderList('empty');
    expect(await screen.findByText('No tasksets yet')).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: 'Create Taskset' })).toHaveLength(2);
  });
});
