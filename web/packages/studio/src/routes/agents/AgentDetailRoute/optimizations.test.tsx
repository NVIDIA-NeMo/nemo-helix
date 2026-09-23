// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_AGENT_OPTIMIZATIONS_ENABLED', 'true');
});

import type { OptimizeJob } from '@nemo/sdk/generated/agents/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { mockOptimizeJobs } from '@studio/mocks/handlers/agentOptimizeJobs';
import { server } from '@studio/mocks/node';
import { AgentDetailRoute } from '@studio/routes/agents/AgentDetailRoute';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { LG_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import { fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const agentName = 'react-agent';
const workspace = workspace1.workspace;

const OPTIMIZE_JOBS_URL = `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/jobs/optimize`;
const OPTIMIZE_JOB_URL = `${OPTIMIZE_JOBS_URL}/:name`;

const FILESET_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name`;

const listOnly = (studyName: string, overrides: Partial<OptimizeJob> = {}) => {
  const data = mockOptimizeJobs
    .filter((job) => job.name === studyName)
    .map((job) => ({ ...job, ...overrides }));
  server.use(
    http.get(OPTIMIZE_JOBS_URL, () =>
      HttpResponse.json({
        data,
        pagination: {
          page: 1,
          page_size: 20,
          current_page_size: data.length,
          total_pages: 1,
          total_results: data.length,
        },
      })
    )
  );
};

const openRowActions = async (user: ReturnType<typeof userEvent.setup>, studyName: string) => {
  const row = await screen.findByRole(
    'row',
    { name: new RegExp(studyName) },
    { timeout: LG_SELECTOR_TIMEOUT }
  );
  await user.click(within(row).getByRole('button', { name: /actions/i }));
  return screen.findByRole('menuitem', { name: 'Delete' });
};

const renderDetail = (search = '?tab=optimizations') =>
  renderRoute(undefined, {
    history: `${getAgentDetailRoute(workspace, agentName)}${search}`,
    routes: [{ path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> }],
  });

describe('AgentDetailRoute optimizations tab', () => {
  it('lists only this agent’s studies', async () => {
    renderDetail();

    expect(await screen.findByRole('tab', { name: 'Optimizations' })).toHaveAttribute(
      'aria-selected',
      'true'
    );
    await waitFor(
      () => {
        expect(screen.getByText('brevity-sweep-3')).toBeInTheDocument();
        expect(screen.getByText('accuracy-sweep-1')).toBeInTheDocument();
        expect(screen.queryByText('other-agent-sweep')).not.toBeInTheDocument();
      },
      { timeout: LG_SELECTOR_TIMEOUT }
    );
  });

  it('opens the launch modal from the Optimize button', async () => {
    renderDetail();

    const optimize = await screen.findByRole('button', { name: 'Optimize' });
    await waitFor(() => expect(optimize).toBeEnabled());
    fireEvent.click(optimize);

    expect(await screen.findByRole('dialog', { name: 'Optimize agent' })).toBeInTheDocument();
  });

  it('scopes the list server-side with a spec.agent filter', async () => {
    const filters: string[] = [];
    const capture = ({ request }: { request: Request }) => {
      const url = new URL(request.url);
      if (!url.pathname.endsWith('/jobs/optimize')) return;
      filters.push(url.searchParams.get('filter') ?? '');
    };
    server.events.on('request:start', capture);

    try {
      renderDetail();
      await screen.findByText('brevity-sweep-3', undefined, { timeout: LG_SELECTOR_TIMEOUT });

      await waitFor(() =>
        expect(filters).toContain(
          JSON.stringify({ 'spec.agent': { $in: [agentName, `${workspace}/${agentName}`] } })
        )
      );
    } finally {
      server.events.removeListener('request:start', capture);
    }
  });

  const captureDeletes = () => {
    const deleted = { studies: [] as string[], filesets: [] as string[] };
    server.use(
      http.delete(OPTIMIZE_JOB_URL, ({ params }) => {
        deleted.studies.push(String(params.name));
        return new HttpResponse(null, { status: 204 });
      }),
      http.delete(FILESET_URL, ({ params }) => {
        deleted.filesets.push(String(params.name));
        return new HttpResponse(null, { status: 204 });
      })
    );
    return deleted;
  };

  const deleteFromRow = async (user: ReturnType<typeof userEvent.setup>, studyName: string) => {
    await user.click(await openRowActions(user, studyName));
    const dialog = await screen.findByRole('dialog', { name: 'Delete Optimization Study' });
    await user.click(within(dialog).getByRole('button', { name: 'Delete' }));
  };

  const BUNDLE = 'react-agent-optimize-abc123';

  /** A study as Studio launches it: its marker names the bundle the study actually runs from. */
  const launchedByStudio = (bundle = BUNDLE) => ({
    spec: {
      optimize_config: 'optimize-brevity.yaml',
      agent: agentName,
      optimize_config_fileset: `${workspace}/${bundle}`,
    },
    custom_fields: { studio_bundle_fileset: bundle },
  });

  /** The files service's view of that bundle, carrying the stamp Studio wrote when it staged it. */
  const serveStampedBundle = () =>
    server.use(
      http.get(FILESET_URL, ({ params }) =>
        HttpResponse.json({
          name: String(params.name),
          custom_fields: { studio_optimize_bundle: agentName },
        })
      )
    );

  it('deletes a finished study from its row actions and leaves filesets alone', async () => {
    const user = userEvent.setup();
    const deleted = captureDeletes();
    listOnly('brevity-sweep-3');
    renderDetail();

    await deleteFromRow(user, 'brevity-sweep-3');

    await waitFor(() => expect(deleted.studies).toEqual(['brevity-sweep-3']));
    expect(deleted.filesets).toEqual([]);
  });

  it('deletes the bundle fileset Studio created for the study', async () => {
    const user = userEvent.setup();
    const deleted = captureDeletes();
    serveStampedBundle();
    listOnly('brevity-sweep-3', launchedByStudio());
    renderDetail();

    await deleteFromRow(user, 'brevity-sweep-3');

    await waitFor(() => expect(deleted.filesets).toEqual([BUNDLE]));
    expect(deleted.studies).toEqual(['brevity-sweep-3']);
  });

  it('leaves a fileset the study does not run from alone, however the study marks it', async () => {
    const user = userEvent.setup();
    const deleted = captureDeletes();
    serveStampedBundle();
    // A marker aimed at someone else's fileset: the study's own spec points somewhere else.
    listOnly('brevity-sweep-3', {
      ...launchedByStudio(),
      custom_fields: { studio_bundle_fileset: 'shared-eval-data' },
    });
    renderDetail();

    await deleteFromRow(user, 'brevity-sweep-3');

    await waitFor(() => expect(deleted.studies).toEqual(['brevity-sweep-3']));
    expect(deleted.filesets).toEqual([]);
  });

  it('keeps the study and names the bundle when the fileset cannot be deleted', async () => {
    const user = userEvent.setup();
    const deleted = captureDeletes();
    serveStampedBundle();
    // Registered after captureDeletes, so this failing handler wins.
    server.use(http.delete(FILESET_URL, () => new HttpResponse(null, { status: 500 })));
    listOnly('brevity-sweep-3', launchedByStudio());
    renderDetail();

    await deleteFromRow(user, 'brevity-sweep-3');

    expect(
      await screen.findByText(new RegExp(BUNDLE), undefined, { timeout: LG_SELECTOR_TIMEOUT })
    ).toBeInTheDocument();
    expect(deleted.studies).toEqual([]);
    expect(await screen.findByText('brevity-sweep-3')).toBeInTheDocument();
  });

  it('does not offer delete while a study is still running', async () => {
    const user = userEvent.setup();
    listOnly('accuracy-sweep-1');
    renderDetail();

    expect(await openRowActions(user, 'accuracy-sweep-1')).toBeDisabled();
  });

  it('offers Optimize from the empty state', async () => {
    const user = userEvent.setup();
    listOnly('no-such-study');
    renderDetail();

    const emptyState = await screen.findByTestId('entity-empty-state-first-use', undefined, {
      timeout: LG_SELECTOR_TIMEOUT,
    });
    await user.click(within(emptyState).getByRole('button', { name: 'Optimize' }));

    expect(await screen.findByRole('dialog', { name: 'Optimize agent' })).toBeInTheDocument();
  });
});
