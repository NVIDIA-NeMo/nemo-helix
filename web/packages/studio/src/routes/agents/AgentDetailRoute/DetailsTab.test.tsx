// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { DetailsTab } from '@studio/routes/agents/AgentDetailRoute/DetailsTab';
import { agentSpecFilesetName } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';
import { getFilesetRoute } from '@studio/routes/utils';
import { render, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const FILESET_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name`;

const baseAgent: Agent = {
  name: 'react-agent',
  workspace: 'default',
  config: {},
  id: 'react-agent',
  created_at: '2026-04-20T10:00:00Z',
  created_by: null,
  updated_at: '2026-04-20T10:00:00Z',
  updated_by: null,
  entity_id: 'react-agent',
  parent: '',
  db_version: 1,
};

describe('DetailsTab', () => {
  it('renders the description as formatted markdown, not raw text', () => {
    render(
      <DetailsTab
        workspace="default"
        agentName="react-agent"
        agent={{ ...baseAgent, description: '# Heading\n\nSee [the docs](https://example.com).' }}
      />
    );

    // The markdown is parsed: a heading becomes a semantic h1 and a link an anchor,
    // rather than showing the raw `#` / `[](...)` characters.
    expect(screen.getByRole('heading', { level: 1, name: 'Heading' })).toBeInTheDocument();
    const link = screen.getByRole('link', { name: 'the docs' });
    expect(link).toHaveAttribute('href', 'https://example.com');
    expect(screen.queryByText(/# Heading/)).not.toBeInTheDocument();
  });

  it('omits the Description row when the agent has no description', () => {
    render(<DetailsTab workspace="default" agentName="react-agent" agent={baseAgent} />);

    expect(screen.queryByText('Description')).not.toBeInTheDocument();
  });

  it('collapses Additional configuration until its header is clicked', async () => {
    const user = userEvent.setup();
    render(
      <DetailsTab
        workspace="default"
        agentName="react-agent"
        agent={{ ...baseAgent, config: { general: { telemetry: 'enabled' } } }}
      />
    );

    expect(screen.getByText('enabled')).not.toBeVisible();

    await user.click(screen.getByText('Additional configuration'));

    expect(screen.getByText('enabled')).toBeVisible();
  });

  it('omits config keys already shown in the Summary panel from Additional configuration', async () => {
    const user = userEvent.setup();
    render(
      <DetailsTab
        workspace="default"
        agentName="react-agent"
        agent={{
          ...baseAgent,
          description: 'Support agent',
          config_format: 'nemo-agents-spec-v1',
          config: {
            config_format: 'nemo-agents-spec-v1',
            name: 'react-agent',
            description: 'Support agent',
            default_harness: 'hermes',
          },
        }}
      />
    );

    await user.click(screen.getByText('Additional configuration'));

    expect(screen.getByText('hermes')).toBeVisible();
    expect(screen.getAllByText('nemo-agents-spec-v1')).toHaveLength(1);
    expect(screen.getAllByText('Support agent')).toHaveLength(1);
  });

  it('links to the agent spec fileset from the Summary panel', async () => {
    server.use(
      http.get(FILESET_URL, ({ params }) =>
        HttpResponse.json({ name: params['name'], workspace: params['workspace'] })
      )
    );
    render(<DetailsTab workspace="default" agentName="react-agent" agent={baseAgent} />);

    const filesetName = agentSpecFilesetName('react-agent');
    const link = await screen.findByRole('link', { name: filesetName });
    expect(link).toHaveAttribute('href', getFilesetRoute('default', filesetName));
  });

  it('omits the Fileset row when the agent has no spec fileset', async () => {
    const requested = vi.fn();
    server.use(
      http.get(FILESET_URL, () => {
        requested();
        return HttpResponse.json({ detail: 'not found' }, { status: 404 });
      })
    );
    render(<DetailsTab workspace="default" agentName="react-agent" agent={baseAgent} />);

    await waitFor(() => expect(requested).toHaveBeenCalled());
    expect(screen.queryByText('Fileset')).not.toBeInTheDocument();
  });
});
