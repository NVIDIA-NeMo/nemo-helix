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
import { getAgentDetailRoute, getAgentRunEvaluationRoute } from '@studio/routes/utils';
import { LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { renderRoute, screen, within } from '@studio/tests/util/render';
import { fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { useNavigate } from 'react-router';

const agentName = 'react-agent';
const workspace = workspace1.workspace;

const renderDetail = () =>
  renderRoute(undefined, {
    history: getAgentDetailRoute(workspace, agentName),
    routes: [{ path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> }],
  });

const PREVIOUS_PAGE = '/previous-page';

const GoBack = () => {
  const navigate = useNavigate();
  return (
    <button type="button" onClick={() => navigate(-1)}>
      Go back
    </button>
  );
};

/**
 * Arrives by link from another page, with the location on screen and a way back, so a test can
 * tell whether the one-shot param was replaced in history rather than pushed over.
 */
const renderArrivingAt = (path: string) =>
  renderRoute(undefined, {
    history: [PREVIOUS_PAGE, path],
    routes: [
      { path: PREVIOUS_PAGE, element: <LocationDisplay /> },
      {
        path: ROUTES.workspace.agentDetail,
        element: (
          <>
            <AgentDetailRoute />
            <LocationDisplay />
            <GoBack />
          </>
        ),
      },
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
        config: {},
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
    expect(screen.getByRole('tab', { name: 'Deployments' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Evaluations' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Logs' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Chat' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Details' })).toBeInTheDocument();
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

    await user.click(await screen.findByRole('tab', { name: 'Details' }));

    expect(await screen.findByRole('button', { name: 'Deploy' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Run Evaluation' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Open traces' })).not.toBeInTheDocument();
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

    await user.click(await screen.findByRole('tab', { name: 'Deployments' }));
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

    await user.click(await screen.findByRole('tab', { name: 'Deployments' }));
    await screen.findByText('Image ready');
    await user.click(screen.getAllByRole('button', { name: /^Deploy$/ })[0]);

    await screen.findByRole('textbox', { name: /Deployment Name/ });
    expect(screen.queryByRole('textbox', { name: 'Container Image' })).not.toBeInTheDocument();
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

  describe('arriving with ?action=run-evaluation', () => {
    it('opens the Run Evaluation modal on the Evaluations tab', async () => {
      renderArrivingAt(getAgentRunEvaluationRoute(workspace, agentName));

      expect(
        await screen.findByRole('dialog', { name: 'Run Agent Evaluation' })
      ).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: 'Evaluations' })).toHaveAttribute(
        'aria-selected',
        'true'
      );
    });

    it('replaces the param in history so reload and Back do not reopen the modal', async () => {
      renderArrivingAt(getAgentRunEvaluationRoute(workspace, agentName));

      await screen.findByRole('dialog', { name: 'Run Agent Evaluation' });
      await waitFor(() =>
        expect(currentLocation()).toBe(
          `${getAgentDetailRoute(workspace, agentName)}?tab=evaluations`
        )
      );

      // Pushed rather than replaced, Back would land on the entry still carrying the action.
      // fireEvent, not userEvent: the open modal makes the page behind it inert to real input.
      fireEvent.click(screen.getByRole('button', { name: 'Go back', hidden: true }));
      await waitFor(() => expect(currentLocation()).toBe(PREVIOUS_PAGE));
    });

    it('lands on the tab without the modal when the agent cannot be evaluated', async () => {
      server.use(
        http.get(`${agentsUrl}/:name`, () =>
          HttpResponse.json({ name: agentName, workspace, created_at: '2026-04-20T10:00:00Z' })
        )
      );
      renderArrivingAt(getAgentRunEvaluationRoute(workspace, agentName));

      await waitFor(() =>
        expect(currentLocation()).toBe(
          `${getAgentDetailRoute(workspace, agentName)}?tab=evaluations`
        )
      );
      expect(screen.getByRole('tab', { name: 'Evaluations' })).toHaveAttribute(
        'aria-selected',
        'true'
      );
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });
  });
});
