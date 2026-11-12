// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { TaskDetailRoute } from '@studio/routes/evaluation/tasks/TaskDetailRoute';
import { renderRoute } from '@studio/tests/util/render';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const renderDetail = (path = 'sort-strings') =>
  renderRoute(undefined, {
    history: `/workspaces/default/evaluation/tasks/${path}`,
    routes: [
      { path: ROUTES.workspace.taskDetail, element: <TaskDetailRoute /> },
      { path: ROUTES.workspace.taskDetailTab, element: <TaskDetailRoute /> },
      { path: ROUTES.workspace.tasksetDetail, element: <div>taskset page</div> },
    ],
  });

describe('TaskDetailRoute', () => {
  it("opens on the overview with an inline task's definition and details", async () => {
    renderDetail();
    expect(
      await screen.findByRole('tab', { name: 'Overview', selected: true })
    ).toBeInTheDocument();
    expect(screen.getByText('Sort a list of strings case-insensitively')).toBeInTheDocument();
    expect(screen.getByText('Inputs')).toBeInTheDocument();
    expect(screen.getByText('Grader only')).toBeInTheDocument();
    expect(screen.getByText('task-sort-strings')).toBeInTheDocument();
    expect(screen.getByText('No metrics score this task.')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /edit|delete/i })).not.toBeInTheDocument();
  });

  it('shows a Harbor task with its environment, metrics, and metadata', async () => {
    renderDetail('sort-ints');
    expect(await screen.findByText('python:3.12-slim')).toBeInTheDocument();
    expect(screen.getByText('timeout_sec')).toBeInTheDocument();
    expect(screen.getByText('Harbor reward')).toBeInTheDocument();
    expect(screen.getByText('v0.1.0')).toBeInTheDocument();
    expect(screen.getByText('difficulty')).toBeInTheDocument();
  });

  it('previews the tasksets that use it and links to the full list', async () => {
    renderDetail();
    const behind = await screen.findByRole('button', { name: /sort-suite/ });
    expect(await within(behind).findByText('Behind latest')).toBeInTheDocument();
    expect(screen.getByText('2 total')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'View all' }));
    expect(screen.getByRole('tab', { name: 'Used By', selected: true })).toBeInTheDocument();
    expect(await screen.findByRole('row', { name: /\bphishing-triage\b/ })).toBeInTheDocument();
  });

  it('shows revision history with the tags on each revision', async () => {
    renderDetail();
    await userEvent.click(await screen.findByRole('tab', { name: 'Revisions' }));
    const r2 = await screen.findByRole('row', { name: /\br2\b/ });
    expect(within(r2).getByText('latest')).toBeInTheDocument();
    expect(screen.getByRole('row', { name: /\br1\b/ })).toBeInTheDocument();
  });

  it('lists the tasksets that use it, flagging a pin behind latest', async () => {
    renderDetail();
    await userEvent.click(await screen.findByRole('tab', { name: 'Used By' }));
    const behind = await screen.findByRole('row', { name: /\bsort-suite\b/ });
    expect(await within(behind).findByText('Behind latest')).toBeInTheDocument();
    expect(within(behind).getByText('r1')).toBeInTheDocument();
    const current = screen.getByRole('row', { name: /\bphishing-triage\b/ });
    expect(within(current).getByText('r2')).toBeInTheDocument();
    expect(within(current).queryByText('Behind latest')).not.toBeInTheDocument();

    await userEvent.click(within(behind).getByText('sort-suite'));
    expect(await screen.findByText('taskset page')).toBeInTheDocument();
  });

  it('opens a tab from its path', async () => {
    renderDetail('sort-strings/revisions');
    expect(
      await screen.findByRole('tab', { name: 'Revisions', selected: true })
    ).toBeInTheDocument();
    expect(await screen.findByRole('row', { name: /\br2\b/ })).toBeInTheDocument();
  });

  it('redirects an unknown tab to the overview', async () => {
    renderDetail('sort-strings/nope');
    expect(
      await screen.findByRole('tab', { name: 'Overview', selected: true })
    ).toBeInTheDocument();
  });

  it('shows an error for an unknown task', async () => {
    renderDetail('missing');
    expect(await screen.findByText(/not found/i)).toBeInTheDocument();
  });
});
