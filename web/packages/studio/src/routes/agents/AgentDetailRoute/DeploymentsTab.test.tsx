// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import type { AgentSpecSource } from '@studio/api/agents/useAgentSpecFileset';
import { DeploymentsTab } from '@studio/routes/agents/AgentDetailRoute/DeploymentsTab';
import { render, screen } from '@studio/tests/util/render';

const STAGED = 'a'.repeat(40);
const MOVED = 'b'.repeat(40);

const source = (revision: string): AgentSpecSource => ({
  owner: 'acme',
  repo: 'agents',
  repository: 'acme/agents',
  trackedRevision: 'main',
  revision,
  webUrl: `https://github.com/acme/agents/tree/${revision}`,
});

const deployment = (overrides: Partial<AgentDeployment> = {}): AgentDeployment =>
  ({ name: 'calc-dep', status: 'running', ...overrides }) as AgentDeployment;

const renderTab = (deployments: AgentDeployment[], specSource?: AgentSpecSource) =>
  render(
    <DeploymentsTab
      agentName="calc"
      deployments={deployments}
      isDeploymentsLoading={false}
      isDeploying={false}
      onDeploy={vi.fn()}
      onChat={vi.fn()}
      onDelete={vi.fn()}
      onViewLogs={vi.fn()}
      canDeploy
      specSource={specSource}
    />
  );

describe('DeploymentsTab staged commit', () => {
  it('links the staged commit to GitHub, opened away from Studio', () => {
    renderTab([deployment({ spec_revision: STAGED })], source(STAGED));

    const link = screen.getByRole('link', { name: 'aaaaaaa' });
    expect(link).toHaveAttribute('href', `https://github.com/acme/agents/commit/${STAGED}`);
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'));
  });

  it('says the source has moved on when the fileset points somewhere newer', () => {
    renderTab([deployment({ spec_revision: STAGED })], source(MOVED));

    expect(screen.getByText(/the source has moved on since/)).toBeInTheDocument();
  });

  it('says nothing about staleness while the deployment is on the current commit', () => {
    renderTab([deployment({ spec_revision: STAGED })], source(STAGED));

    expect(screen.queryByText(/moved on since/)).not.toBeInTheDocument();
  });

  it('names the commit without a link when the source is gone', () => {
    renderTab([deployment({ spec_revision: STAGED })]);

    expect(screen.getByText(/Staged from commit/)).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'aaaaaaa' })).not.toBeInTheDocument();
  });

  it('says nothing for a deployment that staged no fileset', () => {
    renderTab([deployment()], source(STAGED));

    expect(screen.queryByText(/Staged from commit/)).not.toBeInTheDocument();
  });
});
