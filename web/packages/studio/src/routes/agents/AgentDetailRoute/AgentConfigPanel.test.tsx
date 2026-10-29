// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import type { AgentSpecSource } from '@studio/api/agents/useAgentSpecFileset';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { AgentConfigPanel } from '@studio/routes/agents/AgentDetailRoute/AgentConfigPanel';
import { agentSpecFilesetName } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';
import { render, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const FILESET_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name`;
const PINNED = 'a'.repeat(40);

const agent = { name: 'calc', workspace: 'ws', config: { llms: {} } } as unknown as Agent;

const source = (trackedRevision?: string): AgentSpecSource => ({
  owner: 'acme',
  repo: 'agents',
  repository: 'acme/agents',
  trackedRevision,
  revision: PINNED,
  webUrl: `https://github.com/acme/agents/tree/${PINNED}`,
});

const renderPanel = (specSource?: AgentSpecSource) =>
  render(
    <AgentConfigPanel open workspace="ws" agent={agent} specSource={specSource} onClose={vi.fn()} />
  );

describe('AgentConfigPanel', () => {
  it('refreshes the agent spec fileset to the latest commit of its tracked ref', async () => {
    const user = userEvent.setup();
    const refreshes: string[] = [];
    server.use(
      http.post(`${FILESET_URL}/refresh`, ({ params }) => {
        refreshes.push(String(params['name']));
        return HttpResponse.json({
          name: params['name'],
          workspace: 'ws',
          storage: { type: 'github', owner: 'acme', repo: 'agents', revision: PINNED, path: '' },
        });
      })
    );

    renderPanel(source('main'));

    expect(await screen.findByText(/Tracking main at aaaaaaa/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Update to latest' }));

    await waitFor(() => expect(refreshes).toEqual([agentSpecFilesetName('calc')]));
  });

  it('offers no update for a fileset pinned to a commit', async () => {
    renderPanel(source());

    expect(await screen.findByText('Agent config')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Update to latest' })).not.toBeInTheDocument();
  });

  it('offers no update for an agent with no repository source', async () => {
    renderPanel();

    expect(await screen.findByText('Agent config')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Update to latest' })).not.toBeInTheDocument();
  });
});
