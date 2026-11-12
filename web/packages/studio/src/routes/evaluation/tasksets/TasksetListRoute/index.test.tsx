// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { TasksetListRoute } from '@studio/routes/evaluation/tasksets/TasksetListRoute';
import { renderRoute } from '@studio/tests/util/render';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

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

  const openRowMenu = async (name: string) => {
    const row = await screen.findByRole('row', { name: new RegExp(`\\b${name}\\b`) });
    await userEvent.click(within(row).getByRole('button', { name: 'Open quick actions menu' }));
    return within(row).getByRole('menu');
  };

  it('offers the detail page actions on each row, without opening the row', async () => {
    renderList();
    const menu = await openRowMenu('sort-suite');
    const items = within(menu).getAllByRole('menuitem');
    expect(items.map((item) => item.textContent)).toEqual(['Edit', 'Delete']);

    await userEvent.click(within(menu).getByRole('menuitem', { name: 'Edit' }));
    expect(
      await screen.findByRole('dialog', { name: 'Edit Taskset: sort-suite' })
    ).toBeInTheDocument();
    expect(screen.queryByText('detail page')).not.toBeInTheDocument();
  });

  it('deletes a taskset from its row and stays on the list', async () => {
    const deleted = vi.fn();
    server.use(
      http.delete(
        `${PLATFORM_BASE_URL}/apis/evals/v2/workspaces/:workspace/tasksets/:name`,
        ({ params }) => {
          deleted(params.name);
          return new HttpResponse(null, { status: 204 });
        }
      )
    );
    renderList();
    const menu = await openRowMenu('phishing-triage');
    await userEvent.click(within(menu).getByRole('menuitem', { name: 'Delete' }));
    await userEvent.click(await screen.findByRole('button', { name: 'Delete' }));

    await waitFor(() => expect(deleted).toHaveBeenCalledWith('phishing-triage'));
    expect(screen.queryByText('detail page')).not.toBeInTheDocument();
  });
});
