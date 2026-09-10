// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { SourcePanel } from '@studio/routes/agents/AgentDetailRoute/SourcePanel';
import { agentSpecFilesetName } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';
import { render, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { type JsonBodyType, http, HttpResponse } from 'msw';

const FILESET_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name`;
const PINNED = 'a'.repeat(40);
const MOVED = 'b'.repeat(40);

const githubFileset = (revision: string, trackedRevision: string | null = 'main') => ({
  name: 'calc-ethos',
  workspace: 'ws',
  storage: {
    type: 'github',
    owner: 'acme',
    repo: 'agents',
    revision,
    original_revision: trackedRevision,
    path: '',
  },
});

// A refresh repoints the stored fileset, so a later read has to see the new revision.
const mockFileset = (body: JsonBodyType, refreshed?: JsonBodyType) => {
  const refreshes: string[] = [];
  let current = body;
  server.use(
    http.get(FILESET_URL, () =>
      current
        ? HttpResponse.json(current)
        : HttpResponse.json({ detail: 'not found' }, { status: 404 })
    ),
    http.post(`${FILESET_URL}/refresh`, ({ params }) => {
      refreshes.push(String(params['name']));
      current = refreshed ?? current;
      return HttpResponse.json(current);
    })
  );
  return refreshes;
};

const renderPanel = () => render(<SourcePanel workspace="ws" agentName="calc" />);

describe('SourcePanel', () => {
  it('names the repository and the commit the fileset is pinned to', async () => {
    mockFileset(githubFileset(PINNED));

    renderPanel();

    expect(await screen.findByText('acme/agents')).toBeInTheDocument();
    expect(screen.getByText('aaaaaaa')).toBeInTheDocument();
    expect(screen.getByText('main')).toBeInTheDocument();
  });

  it('asks the fileset it belongs to for a newer commit', async () => {
    const user = userEvent.setup();
    const refreshes = mockFileset(githubFileset(PINNED), githubFileset(MOVED));

    renderPanel();
    await user.click(await screen.findByRole('button', { name: 'Update to latest' }));

    await waitFor(() => expect(refreshes).toEqual([agentSpecFilesetName('calc')]));
    expect(await screen.findByText('bbbbbbb')).toBeInTheDocument();
  });

  it('offers no update for a fileset pinned to a commit, which tracks nothing', async () => {
    mockFileset(githubFileset(PINNED, null));

    renderPanel();

    expect(await screen.findByText('acme/agents')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Update to latest' })).not.toBeInTheDocument();
  });

  it('renders nothing for an agent whose files were uploaded, not sourced', async () => {
    mockFileset({ name: 'calc-ethos', workspace: 'ws', storage: { type: 'local', path: '/data' } });

    renderPanel();

    await waitFor(() => expect(screen.queryByText('Source')).not.toBeInTheDocument());
  });

  it('renders nothing for an agent with no fileset at all', async () => {
    mockFileset(null);

    renderPanel();

    await waitFor(() => expect(screen.queryByText('Source')).not.toBeInTheDocument());
  });
});
