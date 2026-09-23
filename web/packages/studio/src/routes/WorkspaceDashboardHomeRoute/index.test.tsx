// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAgentsListAgents } from '@nemo/sdk/generated/agents/agents';
import { useEvaluatorListEvaluateJobs } from '@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes';
import { useInsightsListInsights } from '@nemo/sdk/generated/insights/insights-insights';
import { useListExperiments } from '@nemo/sdk/generated/platform/experiments';
import { useModelsListModels } from '@nemo/sdk/generated/platform/models';
import { ROUTES } from '@studio/constants/routes';
import { WorkspaceDashboardHomeRoute } from '@studio/routes/WorkspaceDashboardHomeRoute';
import { queryResult } from '@studio/routes/WorkspaceDashboardHomeRoute/testMocks';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { fireEvent, render, screen } from '@testing-library/react';
import { createMemoryRouter, generatePath, RouterProvider } from 'react-router';

vi.mock('@nemo/sdk/generated/agents/agents', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/agents/agents')>()),
  useAgentsListAgents: vi.fn(),
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

const renderRoute = () => {
  const dashboardPath = generatePath(ROUTES.workspace.dashboard, { workspace: TEST_WORKSPACE });

  const router = createMemoryRouter(
    [{ path: ROUTES.workspace.dashboard, element: <WorkspaceDashboardHomeRoute /> }],
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
    window.localStorage.clear();
    vi.mocked(useAgentsListAgents).mockReturnValue(queryResult(1));
    vi.mocked(useInsightsListInsights).mockReturnValue(queryResult(4));
    vi.mocked(useEvaluatorListEvaluateJobs).mockReturnValue(queryResult(120));
    vi.mocked(useListExperiments).mockReturnValue(queryResult(1));
    vi.mocked(useModelsListModels).mockReturnValue(queryResult(0));
  });

  it('renders the page header, stat tiles, and quickstart panels', async () => {
    renderRoute();

    expect(await screen.findByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Agents')).toBeInTheDocument();
    expect(screen.getByText('Quickstart')).toBeInTheDocument();
    expect(screen.getByText('Connect an Agent')).toBeInTheDocument();
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
});
