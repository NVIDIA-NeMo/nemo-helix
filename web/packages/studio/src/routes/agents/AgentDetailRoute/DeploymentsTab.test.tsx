// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import type { AgentSpecSource } from '@studio/api/agents/useAgentSpecFileset';
import { DeploymentsTab } from '@studio/routes/agents/AgentDetailRoute/DeploymentsTab';
import { renderRoute, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

const STAGED = 'a'.repeat(40);
const MOVED = 'b'.repeat(40);

const LONG_ERROR =
  "No container-reachable inference base URL for k8s deployment: platform base URL 'http://127.0.0.1:8080' is not usable from an agent pod and no internal API Service URL is set. Set NEMO_INTERNAL_BASE_URL / NMP_INTERNAL_BASE_URL (or deploy with a cluster-internal gateway address).";

const failedDeployment = {
  name: 'calculator-agent-2-17aa2130',
  workspace: 'default',
  status: 'failed',
  error: LONG_ERROR,
} as AgentDeployment;

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
  renderRoute(
    <DeploymentsTab
      agentName="calculator-agent"
      deployments={deployments}
      isDeploymentsLoading={false}
      isDeploying={false}
      onDeploy={vi.fn()}
      onChat={vi.fn()}
      onDelete={vi.fn()}
      onViewLogs={vi.fn()}
      canDeploy
      specSource={specSource}
      workspace="default"
      canPackage
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

  it('names the newer commit the source points at', () => {
    renderTab([deployment({ spec_revision: STAGED })], source(MOVED));

    expect(screen.getByText(/source is now/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'bbbbbbb' })).toHaveAttribute(
      'href',
      `https://github.com/acme/agents/commit/${MOVED}`
    );
  });

  it('says nothing about the source while the deployment is on its current commit', () => {
    renderTab([deployment({ spec_revision: STAGED })], source(STAGED));

    expect(screen.queryByText(/source is now/)).not.toBeInTheDocument();
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

describe('DeploymentsTab', () => {
  it('makes a long failure message readable instead of ellipsising it away', async () => {
    const user = userEvent.setup();
    renderTab([failedDeployment]);

    const toggle = screen.getByRole('button', { name: 'Show full error' });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');

    await user.click(toggle);

    const expanded = screen.getByRole('button', { name: 'Show less' });
    expect(expanded).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText(LONG_ERROR)).toBeInTheDocument();
  });

  it.each([
    ['docker', 'Docker'],
    ['k8s', 'Kubernetes'],
    ['subprocess', 'Subprocess'],
  ])('names a %s deployment its runtime, since a row otherwise hides it', (mode, label) => {
    renderTab([deployment({ deployment_mode: mode as AgentDeployment['deployment_mode'] })]);

    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it('calls a deployment with no recorded runtime a subprocess, which is the default', () => {
    renderTab([deployment()]);

    expect(screen.getByText('Subprocess')).toBeInTheDocument();
  });

  it('offers packaging from the deployments header', () => {
    renderTab([failedDeployment]);

    expect(screen.getByRole('button', { name: /Build image|Manage image/ })).toBeInTheDocument();
  });

  it('names the image a deployment is running, since an agent has many over time', () => {
    renderTab([
      {
        name: 'calculator-agent-1',
        workspace: 'default',
        status: 'running',
        image: 'nemo-agents/default/calculator-agent-9594db954f89:26.09.04',
      } as AgentDeployment,
    ]);

    expect(
      screen.getByText('nemo-agents/default/calculator-agent-9594db954f89:26.09.04')
    ).toBeInTheDocument();
  });

  it('offers no error toggle when a deployment has not failed', () => {
    renderTab([
      {
        name: 'calculator-agent-1',
        workspace: 'default',
        status: 'running',
        endpoint: 'http://localhost:9001',
      } as AgentDeployment,
    ]);

    expect(screen.queryByRole('button', { name: /error/i })).not.toBeInTheDocument();
  });
});
