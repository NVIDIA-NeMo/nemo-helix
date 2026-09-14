// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_INTAKE_ENABLED', 'true');
  vi.stubEnv('VITE_FF_AGENT_OVERVIEW_ENABLED', 'true');
});

import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { AgentDetailRoute } from '@studio/routes/agents/AgentDetailRoute';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen } from '@studio/tests/util/render';
import { waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const agentName = 'react-agent';
const workspace = workspace1.workspace;

const renderDetail = () =>
  renderRoute(undefined, {
    history: getAgentDetailRoute(workspace, agentName),
    routes: [
      { path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> },
      { path: ROUTES.workspace.intakeTraces, element: <div>intake-traces-page</div> },
    ],
  });

describe('AgentDetailRoute', () => {
  it('renders the agent as a full page with tabs and header actions', async () => {
    renderDetail();

    expect(await screen.findByTestId('nv-page-header-heading')).toHaveTextContent(agentName);
    expect(screen.getByRole('tab', { name: 'Overview' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: 'Deployments' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Evaluations' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Logs' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Chat' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Details' })).toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: 'Configuration' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open traces' })).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getAllByRole('button', { name: 'Run evaluation' })).toHaveLength(2);
    });
    expect(screen.getByRole('button', { name: 'Deploy' })).toBeInTheDocument();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('lands on the overview tab with trace statistics and the details panel', async () => {
    renderDetail();

    expect(await screen.findByText('Trace statistics')).toBeInTheDocument();
    expect(screen.getByText('Agent ID')).toBeInTheDocument();
    expect(screen.getByText('Created')).toBeInTheDocument();
  });

  it('navigates to intake traces when Open traces is clicked', async () => {
    const user = userEvent.setup();
    renderDetail();

    await user.click(await screen.findByRole('button', { name: 'Open traces' }));

    expect(await screen.findByText('intake-traces-page')).toBeInTheDocument();
  });

  it('narrows the header to a single primary action off the overview tab', async () => {
    const user = userEvent.setup();
    renderDetail();

    await user.click(await screen.findByRole('tab', { name: 'Details' }));

    expect(await screen.findByRole('button', { name: 'Deploy' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Run evaluation' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Open traces' })).not.toBeInTheDocument();
  });

  it('switches to the chat tab', async () => {
    const user = userEvent.setup();
    renderDetail();

    await user.click(await screen.findByRole('tab', { name: 'Chat' }));

    expect(screen.getByRole('tab', { name: 'Chat' })).toHaveAttribute('aria-selected', 'true');
    expect(await screen.findByRole('textbox', { name: /Task prompt/i })).toBeInTheDocument();
  });

  it('shows the agent spec on the details tab and masks secrets', async () => {
    const user = userEvent.setup();
    renderDetail();

    await user.click(await screen.findByRole('tab', { name: 'Details' }));

    expect(screen.getByRole('tab', { name: 'Details' })).toHaveAttribute('aria-selected', 'true');
    // Structured panels
    expect(await screen.findByText('Summary')).toBeInTheDocument();
    expect(screen.getByText('Workflow')).toBeInTheDocument();
    expect(screen.getByText('Models')).toBeInTheDocument();
    expect(screen.getByText('Tools')).toBeInTheDocument();
    // Config values surfaced from the spec
    expect(screen.getByText('nat-workflow-v1')).toBeInTheDocument();
    expect(screen.getByText('react_agent')).toBeInTheDocument();
    // The llm api_key is masked, never shown raw
    expect(screen.queryByText('not-used')).not.toBeInTheDocument();
    expect(screen.getByText('••••••••')).toBeInTheDocument();
  });
});
