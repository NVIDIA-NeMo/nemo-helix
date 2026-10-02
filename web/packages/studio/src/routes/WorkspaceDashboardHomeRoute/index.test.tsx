// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { useAgentsListDeployments } from '@nemo/sdk/generated/agents/agent-deployments';
import { useAgentsGetAgent, useAgentsListAgents } from '@nemo/sdk/generated/agents/agents';
import { useEvaluatorListEvaluateJobs } from '@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes';
import { useInsightsListInsights } from '@nemo/sdk/generated/insights/insights-insights';
import { useListExperiments } from '@nemo/sdk/generated/platform/experiments';
import { useModelsListModels } from '@nemo/sdk/generated/platform/models';
import { ROUTES } from '@studio/constants/routes';
import { getWorkspaceDetailsDefaultRoute } from '@studio/routes/utils';
import { WorkspaceDashboardHomeRoute } from '@studio/routes/WorkspaceDashboardHomeRoute';
import { queryResult } from '@studio/routes/WorkspaceDashboardHomeRoute/testMocks';
import {
  SAMPLE_AGENT_NAME,
  SAMPLE_WORKSPACE,
} from '@studio/routes/WorkspaceDashboardHomeRoute/useSampleQuickstartAgent';
import { LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, generatePath, RouterProvider } from 'react-router';

vi.mock('@nemo/sdk/generated/agents/agents', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/agents/agents')>()),
  useAgentsListAgents: vi.fn(),
  useAgentsGetAgent: vi.fn(),
}));
vi.mock('@nemo/sdk/generated/agents/agent-deployments', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/agents/agent-deployments')>()),
  useAgentsListDeployments: vi.fn(),
}));
vi.mock('@nemo/sdk/generated/insights/insights-insights', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/insights/insights-insights')>()),
  useInsightsListInsights: vi.fn(),
}));
vi.mock('@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes', async (importOriginal) => ({
  ...(await importOriginal<
    typeof import('@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes')
  >()),
  useEvaluatorListEvaluateJobs: vi.fn(),
}));
vi.mock('@nemo/sdk/generated/platform/experiments', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/experiments')>()),
  useListExperiments: vi.fn(),
}));
vi.mock('@nemo/sdk/generated/platform/models', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/models')>()),
  useModelsListModels: vi.fn(),
}));

const TEST_WORKSPACE = 'test-workspace';

const renderRoute = (workspace = TEST_WORKSPACE) => {
  const dashboardPath = generatePath(ROUTES.workspace.dashboard, { workspace });

  const router = createMemoryRouter(
    [
      {
        path: ROUTES.workspace.dashboard,
        element: (
          <>
            <WorkspaceDashboardHomeRoute />
            <LocationDisplay />
          </>
        ),
      },
      // Wherever a dashboard link leads, so a test can see it was followed.
      { path: '*', element: <LocationDisplay /> },
    ],
    { initialEntries: [dashboardPath] }
  );

  return render(
    <TestProviders>
      <RouterProvider router={router} />
    </TestProviders>
  );
};

describe('WorkspaceDashboardHomeRoute', () => {
  beforeEach(() => {
    // Call history too, so assertions about how a hook was called see only this test's render.
    vi.clearAllMocks();
    window.localStorage.clear();
    vi.mocked(useAgentsListAgents).mockReturnValue(queryResult(1));
    vi.mocked(useInsightsListInsights).mockReturnValue(queryResult(4));
    vi.mocked(useEvaluatorListEvaluateJobs).mockReturnValue(queryResult(120));
    vi.mocked(useListExperiments).mockReturnValue(queryResult(1));
    vi.mocked(useModelsListModels).mockReturnValue(queryResult(0));
    vi.mocked(useAgentsGetAgent).mockReturnValue({
      data: { name: SAMPLE_AGENT_NAME, description: 'Sample email triage agent.' },
      isFetched: true,
    } as never);
    vi.mocked(useAgentsListDeployments).mockReturnValue({
      data: {
        data: [
          { name: `${SAMPLE_AGENT_NAME}-0000aaaa`, agent: SAMPLE_AGENT_NAME, status: 'failed' },
          { name: `${SAMPLE_AGENT_NAME}-9f2a1c00`, agent: SAMPLE_AGENT_NAME, status: 'running' },
          { name: 'other-agent-12345678', agent: 'other-agent', status: 'running' },
        ],
      },
      isFetched: true,
    } as never);
  });

  it('renders the page header, stat tiles, and quickstart panels', async () => {
    renderRoute();

    expect(await screen.findByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Agents')).toBeInTheDocument();
    expect(screen.getByText('Quickstart')).toBeInTheDocument();
    expect(screen.getByText('Connect an Agent')).toBeInTheDocument();
  });

  it('shows the sample sandbox banner only in the sample workspace', async () => {
    const { unmount } = renderRoute('sample');

    expect(await screen.findByText(/Sample Sandbox\./)).toBeInTheDocument();
    unmount();

    renderRoute();
    expect(await screen.findByText('Dashboard')).toBeInTheDocument();
    expect(screen.queryByText(/Sample Sandbox\./)).not.toBeInTheDocument();
  });

  it('marks the get-started area for the Welcome Tour', async () => {
    renderRoute();

    expect(await screen.findByText('Agents')).toBeInTheDocument();
    // The Welcome Tour matches this selector directly, not via a Testing Library query.
    // eslint-disable-next-line testing-library/no-node-access
    expect(document.querySelector('[data-tour="dashboard-get-started"]')).toBeInTheDocument();
  });

  it('moves focus to the get-started area instead of dropping it when Quickstart is dismissed', async () => {
    renderRoute();

    await screen.findByText('Quickstart');
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss Quickstart' }));

    expect(screen.queryByText('Quickstart')).not.toBeInTheDocument();
    // Focus management has no Testing Library query equivalent — raw DOM access is required.
    /* eslint-disable testing-library/no-node-access */
    const getStartedElement = document.querySelector('[data-tour="dashboard-get-started"]');
    expect(document.activeElement).toBe(getStartedElement);
    expect(document.activeElement).not.toBe(document.body);
    /* eslint-enable testing-library/no-node-access */
  });

  it('does not fetch the sample agent outside the sample workspace', async () => {
    renderRoute();

    expect(await screen.findByText('Connect an Agent')).toBeInTheDocument();
    expect(screen.queryByTestId('quickstart-sample-agent-row')).not.toBeInTheDocument();
    expect(vi.mocked(useAgentsGetAgent)).toHaveBeenLastCalledWith(
      TEST_WORKSPACE,
      SAMPLE_AGENT_NAME,
      expect.objectContaining({ query: expect.objectContaining({ enabled: false }) })
    );
    // The deployments query polls, so leaving it on would poll in every workspace.
    expect(vi.mocked(useAgentsListDeployments)).toHaveBeenLastCalledWith(
      TEST_WORKSPACE,
      expect.anything(),
      expect.objectContaining({ query: expect.objectContaining({ enabled: false }) })
    );
  });

  describe('in the sample workspace', () => {
    it('renders the QuickstartSamplePanel instead of the QuickstartSection', async () => {
      renderRoute(SAMPLE_WORKSPACE);

      expect(await screen.findByTestId('quickstart-sample-agent-row')).toBeInTheDocument();
      expect(screen.getByText('Quickstart')).toBeInTheDocument();
      expect(
        screen.getByText(
          'A sample workload with an agent and dataset already loaded. Inspect what shipped, or run any step yourself.'
        )
      ).toBeInTheDocument();
      expect(screen.getByText(SAMPLE_AGENT_NAME)).toBeInTheDocument();
      expect(screen.getByText('Sample email triage agent.')).toBeInTheDocument();
      expect(screen.queryByText('Connect an Agent')).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: 'Dismiss Quickstart' })).not.toBeInTheDocument();
    });

    it('switches to the shared workspace from the footer', async () => {
      const user = userEvent.setup();
      renderRoute(SAMPLE_WORKSPACE);

      expect(await screen.findByText('Ready to start with your own assets?')).toBeInTheDocument();
      await user.click(screen.getByRole('button', { name: 'Switch to Shared Workspace' }));

      await waitFor(() =>
        expect(screen.getByTestId(LOCATION_DISPLAY_TEST_ID).textContent).toBe(
          getWorkspaceDetailsDefaultRoute(DEFAULT_WORKSPACE)
        )
      );
    });

    it("shows the sample agent's running deployment in the CLI commands", async () => {
      const user = userEvent.setup();
      renderRoute(SAMPLE_WORKSPACE);

      expect(await screen.findByText('Running')).toBeInTheDocument();
      await user.click(screen.getByRole('radio', { name: 'NeMo CLI' }));

      // Code snippets load asynchronously; the first one is step 1's chat command.
      const [chatCommand] = await screen.findAllByTestId('nv-code-snippet-code');
      expect(chatCommand.textContent).toBe(
        [
          'nemo agents chat',
          `--agent-deployment '${SAMPLE_AGENT_NAME}-9f2a1c00'`,
          "--input 'Hello agent!'",
          `--workspace '${SAMPLE_WORKSPACE}'`,
        ].join(' \\\n  ')
      );
    });

    it('holds the panel back until the deployments load, rather than showing a placeholder', async () => {
      vi.mocked(useAgentsListDeployments).mockReturnValue({
        data: undefined,
        isFetched: false,
      } as never);

      renderRoute(SAMPLE_WORKSPACE);

      expect(await screen.findByText('Agents')).toBeInTheDocument();
      expect(screen.queryByTestId('quickstart-sample-agent-row')).not.toBeInTheDocument();
      // Nor the regular Quickstart, which would flash and then swap out.
      expect(screen.queryByText('Connect an Agent')).not.toBeInTheDocument();
    });

    it('shows neither Quickstart while the sample agent is still loading', async () => {
      vi.mocked(useAgentsGetAgent).mockReturnValue({ data: undefined, isFetched: false } as never);

      renderRoute(SAMPLE_WORKSPACE);

      expect(await screen.findByText('Agents')).toBeInTheDocument();
      expect(screen.queryByText('Quickstart')).not.toBeInTheDocument();
    });

    it('falls back to the regular Quickstart when there is no sample agent to show', async () => {
      // Settled without data: a 404 from a hand-made `sample` workspace, a 403, or a 5xx.
      vi.mocked(useAgentsGetAgent).mockReturnValue({ data: undefined, isFetched: true } as never);

      renderRoute(SAMPLE_WORKSPACE);

      expect(await screen.findByText('Connect an Agent')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Dismiss Quickstart' })).toBeInTheDocument();
      expect(screen.queryByTestId('quickstart-sample-agent-row')).not.toBeInTheDocument();
    });
  });
});
