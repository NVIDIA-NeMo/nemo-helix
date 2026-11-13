// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { TasksetDetailRoute } from '@studio/routes/evaluation/tasksets/TasksetDetailRoute';
import { renderRoute } from '@studio/tests/util/render';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const EVALS_URL = `${PLATFORM_BASE_URL}/apis/evals/v2/workspaces/:workspace`;

const renderDetail = (path = '/workspaces/default/evaluation/tasksets/sort-suite') =>
  renderRoute(undefined, {
    history: path,
    routes: [{ path: ROUTES.workspace.tasksetDetail, element: <TasksetDetailRoute /> }],
  });

describe('TasksetDetailRoute', () => {
  it('summarizes the taskset', async () => {
    renderDetail();
    expect(await screen.findByText('Sorting algorithm tasks')).toBeInTheDocument();
    expect(screen.getByText('r2')).toBeInTheDocument();
    expect(screen.getByText('latest, stable')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /edit|delete/i })).not.toBeInTheDocument();
  });

  it('lists member tasks by runner, and flags a ref that no longer resolves', async () => {
    renderDetail();
    const intsRow = await screen.findByRole('row', { name: /\bsort-ints\b/ });
    expect(await within(intsRow).findByText('Harbor')).toBeInTheDocument();
    expect(within(intsRow).getByText('difficulty: easy')).toBeInTheDocument();
    const stringsRow = screen.getByRole('row', { name: /\bsort-strings\b/ });
    expect(await within(stringsRow).findByText('Inline')).toBeInTheDocument();
    const stableRow = screen.getByRole('row', { name: /\bsort-stable\b/ });
    expect(await within(stableRow).findByText('Not found')).toBeInTheDocument();
  });

  it('opens a Harbor task with its instruction, environment, and tests', async () => {
    renderDetail();
    await userEvent.click(await screen.findByText('sort-ints'));
    const drawer = await screen.findByRole('dialog');
    expect(await within(drawer).findByText(/Sort the integers in input.txt/)).toBeInTheDocument();
    expect(within(drawer).getByText('python:3.12-slim')).toBeInTheDocument();
    expect(within(drawer).getByText('timeout_sec')).toBeInTheDocument();
    expect(
      within(drawer).getByRole('link', { name: 'default/sort-suite-files#tasks/sort-ints.tar.gz' })
    ).toBeInTheDocument();
  });

  it('opens an inline task from a deep link with its inputs and reference', async () => {
    renderDetail('/workspaces/default/evaluation/tasksets/sort-suite?task=default%2Fsort-strings');
    const drawer = await screen.findByRole('dialog');
    expect(
      await within(drawer).findByText('Sort a list of strings case-insensitively')
    ).toBeInTheDocument();
    expect(within(drawer).getByText('Inputs')).toBeInTheDocument();
    expect(within(drawer).getByText('Reference')).toBeInTheDocument();
  });

  it('flags only the rows of a failed batch, keeping the rest resolved', async () => {
    server.use(
      http.get(`${EVALS_URL}/tasksets/:name`, () =>
        HttpResponse.json({
          id: 'taskset-mixed',
          name: 'sort-suite',
          workspace: 'default',
          revision: 1,
          tasks: ['default/sort-ints', 'staging/smoke'],
          created_at: '2026-10-01T09:00:00Z',
          updated_at: '2026-10-01T09:00:00Z',
        })
      ),
      // Returning nothing falls through to the default handler for other workspaces.
      http.get(`${EVALS_URL}/tasks`, ({ params }) =>
        params.workspace === 'staging'
          ? HttpResponse.json({ detail: 'boom' }, { status: 500 })
          : undefined
      )
    );
    renderDetail();
    const intsRow = await screen.findByRole('row', { name: /\bsort-ints\b/ });
    expect(await within(intsRow).findByText('Harbor')).toBeInTheDocument();
    const smokeRow = screen.getByRole('row', { name: /staging\/smoke/ });
    expect(await within(smokeRow).findByText('Failed to load')).toBeInTheDocument();
  });

  it('opens a pinned ref at its pinned revision, not the latest', async () => {
    renderDetail(
      '/workspaces/default/evaluation/tasksets/sort-suite?task=default%2Fsort-ints%23v1'
    );
    const drawer = await screen.findByRole('dialog');
    expect(await within(drawer).findByText('r1')).toBeInTheDocument();
    expect(within(drawer).getByText(/Sort the integers in input.txt/)).toBeInTheDocument();
  });

  it('shows revision history with the tags on each revision', async () => {
    renderDetail();
    await userEvent.click(await screen.findByRole('tab', { name: 'Revisions' }));
    const r1 = await screen.findByRole('row', { name: /\br1\b/ });
    expect(within(r1).getByText('stable')).toBeInTheDocument();
  });

  it('shows an error for an unknown taskset', async () => {
    renderDetail('/workspaces/default/evaluation/tasksets/missing');
    expect(await screen.findByText(/not found/i)).toBeInTheDocument();
  });
});
