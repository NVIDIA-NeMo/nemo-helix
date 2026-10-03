// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import { AgentSummaryPanel } from '@studio/routes/agents/AgentDetailRoute/overview/AgentSummaryPanel';
import { render, screen } from '@studio/tests/util/render';

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

describe('AgentSummaryPanel', () => {
  it('renders the description as formatted markdown, not raw text', () => {
    render(
      <AgentSummaryPanel
        agent={{ ...baseAgent, description: '# Heading\n\nSee [the docs](https://example.com).' }}
        modelNames={[]}
      />
    );

    expect(screen.getByRole('heading', { level: 1, name: 'Heading' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'the docs' })).toHaveAttribute(
      'href',
      'https://example.com'
    );
    expect(screen.queryByText(/# Heading/)).not.toBeInTheDocument();
  });

  it('omits the Description row when the agent has no description', () => {
    render(<AgentSummaryPanel agent={baseAgent} modelNames={[]} />);

    expect(screen.queryByText('Description')).not.toBeInTheDocument();
  });
});
