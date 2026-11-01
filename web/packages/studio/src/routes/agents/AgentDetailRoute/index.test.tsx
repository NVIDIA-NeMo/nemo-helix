// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_INTAKE_ENABLED', 'true');
  vi.stubEnv('VITE_FF_AGENT_OVERVIEW_ENABLED', 'true');
  vi.stubEnv('VITE_FF_AGENT_CONTAINER_DEPLOYMENTS_ENABLED', 'true');
});

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { AgentDetailRoute } from '@studio/routes/agents/AgentDetailRoute';
import {
  getAgentDetailRoute,
  getAgentEvaluationsTabRoute,
  getAgentRunEvaluationRoute,
} from '@studio/routes/utils';
import { LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { renderRoute, screen, within } from '@studio/tests/util/render';
import { waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { delay, http, HttpResponse } from 'msw';

const agentName = 'react-agent';
const workspace = workspace1.workspace;

const renderDetail = () =>
  renderRoute(undefined, {
    history: getAgentDetailRoute(workspace, agentName),
    routes: [{ path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> }],
  });

/** The agent page, plus a stand-in for the Run Agent Evaluation page that reports where it is. */
const renderWithWizardRoute = () =>
  renderRoute(undefined, {
    history: getAgentDetailRoute(workspace, agentName),
    routes: [
      { path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> },
      { path: ROUTES.workspace.agentEvaluationNew, element: <LocationDisplay /> },
    ],
  });

const currentLocation = () => screen.getByTestId(LOCATION_DISPLAY_TEST_ID).textContent;

const BUILT_IMAGE = 'nemo-agents/default/react-agent:1.0';
const agentsUrl = '*/apis/agents/v2/workspaces/:workspace/agents';
const jobsUrl = '*/apis/agents/v2/workspaces/:workspace/jobs/package';

/** A Fabric agent whose most recent packaging job already produced BUILT_IMAGE. */
const mockPreviouslyPackagedAgent = () => {
  server.use(
    http.get(`${agentsUrl}/:name`, () =>
      HttpResponse.json({
        name: agentName,
        workspace,
        description: '',
        created_at: '2026-04-20T10:00:00Z',
        config: { models: { default: { model: 'nvidia/test-model' } } },
        config_format: 'nemo-agents-spec-v1',
      })
    ),
    http.get(jobsUrl, () =>
      HttpResponse.json({ data: [{ name: 'pkg-1', spec: { agent: agentName } }], total: 1 })
    ),
    http.get(`${jobsUrl}/:name/status`, () => HttpResponse.json({ status: 'completed' })),
    http.get(`${jobsUrl}/:name/logs`, () => HttpResponse.json({ data: [], next_page: null })),
    http.get(`${jobsUrl}/:name/results/package_result/download`, () =>
      HttpResponse.json({ image: BUILT_IMAGE, agent: agentName, published: '' })
    )
  );
};

describe('AgentDetailRoute', () => {
  it('renders the agent as a full page with tabs and header actions', async () => {
    renderDetail();

    expect(await screen.findByTestId('nv-page-header-heading')).toHaveTextContent(agentName);
    expect(screen.getByRole('tab', { name: 'Overview' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: 'Deployment' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Evaluations' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Logs' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Chat' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Agent config' })).toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: 'Configuration' })).not.toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getAllByRole('button', { name: 'Run Evaluation' })).toHaveLength(2);
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

  it('narrows the header to a single primary action off the overview tab', async () => {
    const user = userEvent.setup();
    renderDetail();

    await user.click(await screen.findByRole('tab', { name: 'Deployment' }));

    expect(await screen.findByRole('button', { name: 'Deploy' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Run Evaluation' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Open traces' })).not.toBeInTheDocument();
  });

  it('offers Upload instead of Deploy for an agent registered from traces (empty config)', async () => {
    const user = userEvent.setup();
    server.use(
      http.get(`${agentsUrl}/:name`, () =>
        HttpResponse.json({
          name: agentName,
          workspace,
          description: '',
          created_at: '2026-04-20T10:00:00Z',
          config: {},
        })
      )
    );
    server.use(
      http.get('*/apis/agents/v2/workspaces/:workspace/deployments', () =>
        HttpResponse.json({ data: [], total: 0 })
      )
    );
    renderDetail();

    await user.click(await screen.findByRole('tab', { name: 'Deployment' }));

    expect(await screen.findByText(/does not have a configuration/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Deploy' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Deploy this Agent' })).not.toBeInTheDocument();

    await user.click(screen.getAllByRole('button', { name: 'Upload' })[0]!);

    expect(await screen.findByDisplayValue(`${agentName}-v2`)).toBeInTheDocument();
  });

  it('shows the loading state, with no actions, until the agent loads', async () => {
    server.use(http.get(`${agentsUrl}/:name`, () => delay('infinite')));
    renderDetail();

    expect(await screen.findByRole('status', { name: 'Loading...' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Deploy' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Upload' })).not.toBeInTheDocument();
  });

  it('switches to the chat tab', async () => {
    const user = userEvent.setup();
    renderDetail();

    await user.click(await screen.findByRole('tab', { name: 'Chat' }));

    expect(screen.getByRole('tab', { name: 'Chat' })).toHaveAttribute('aria-selected', 'true');
    expect(await screen.findByRole('textbox', { name: /Task prompt/i })).toBeInTheDocument();
  });

  it("offers this agent's built image to a deployment when asked for it", async () => {
    mockPreviouslyPackagedAgent();
    renderDetail();
    const user = userEvent.setup();

    await user.click(await screen.findByRole('tab', { name: 'Deployment' }));
    // The tag lives in the packaging modal now; the trigger is what reports it is ready.
    await screen.findByText('Image ready');
    await user.click(screen.getByRole('button', { name: /Manage image/ }));
    const dialog = await screen.findByRole('dialog');
    await user.click(within(dialog).getByRole('button', { name: /^Deploy$/ }));

    expect(await screen.findByRole('textbox', { name: 'Container Image' })).toHaveValue(
      BUILT_IMAGE
    );
  });

  it('does not make an image built earlier the silent default for a new deployment', async () => {
    mockPreviouslyPackagedAgent();
    renderDetail();
    const user = userEvent.setup();

    await user.click(await screen.findByRole('tab', { name: 'Deployment' }));
    await screen.findByText('Image ready');
    await user.click(screen.getAllByRole('button', { name: /^Deploy$/ })[0]);

    await screen.findByRole('textbox', { name: /Deployment Name/ });
    expect(screen.queryByRole('textbox', { name: 'Container Image' })).not.toBeInTheDocument();
  });

  it('opens the agent config as masked JSON from the settings button', async () => {
    const user = userEvent.setup();
    renderDetail();

    await screen.findByText('nat-workflow-v1');
    await user.click(screen.getByRole('button', { name: 'Agent config' }));

    const dialog = await screen.findByRole('dialog');
    await waitFor(() => expect(dialog).toHaveTextContent('react_agent'));
    expect(dialog).toHaveTextContent('••••••••');
    expect(dialog).not.toHaveTextContent('not-used');
  });

  it("shows the agent's description in the overview Details panel, not the page header", async () => {
    const description = 'A long agent description that would otherwise bloat the header row.';
    server.use(
      http.get(
        `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents/:name`,
        ({ params }) =>
          HttpResponse.json({
            name: params['name'],
            workspace: params['workspace'],
            description,
            config: {},
            config_format: 'nat-workflow-v1',
          })
      )
    );

    renderDetail();

    const descriptionEl = await screen.findByText(description);
    expect(descriptionEl).toBeInTheDocument();
    // The header carries the name and status only - no description, no model.
    expect(
      within(screen.getByTestId('nv-page-header-heading')).queryByText(description)
    ).not.toBeInTheDocument();
  });

  it('opens the Run Agent Evaluation page from the header', async () => {
    const user = userEvent.setup();
    renderWithWizardRoute();

    const [headerButton] = await screen.findAllByRole('button', { name: 'Run Evaluation' });
    await waitFor(() => expect(headerButton).toBeEnabled());
    await user.click(headerButton);

    await waitFor(() =>
      expect(currentLocation()).toBe(getAgentRunEvaluationRoute(workspace, agentName))
    );
  });

  it('sends an old ?action=run-evaluation link to the Run Agent Evaluation page', async () => {
    renderRoute(undefined, {
      history: `${getAgentEvaluationsTabRoute(workspace, agentName)}&action=run-evaluation`,
      routes: [
        { path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> },
        { path: ROUTES.workspace.agentEvaluationNew, element: <LocationDisplay /> },
      ],
    });

    await waitFor(() =>
      expect(currentLocation()).toBe(getAgentRunEvaluationRoute(workspace, agentName))
    );
  });
});
