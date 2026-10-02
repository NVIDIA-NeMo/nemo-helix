// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAgentsListAgents } from '@nemo/sdk/generated/agents/agents';
import { useEvaluatorListEvaluateJobs } from '@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes';
import { useInsightsListInsights } from '@nemo/sdk/generated/insights/insights-insights';
import { useListExperiments } from '@nemo/sdk/generated/platform/experiments';
import { useModelsListModels } from '@nemo/sdk/generated/platform/models';
import { StatTileRow } from '@studio/routes/WorkspaceDashboardHomeRoute/StatTileRow';
import { queryResult } from '@studio/routes/WorkspaceDashboardHomeRoute/testMocks';
import { render, screen } from '@testing-library/react';
import type { ComponentProps } from 'react';
import { MemoryRouter } from 'react-router';

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

const mockUseAgentsListAgents = vi.mocked(useAgentsListAgents);
const mockUseInsightsListInsights = vi.mocked(useInsightsListInsights);
const mockUseEvaluatorListEvaluateJobs = vi.mocked(useEvaluatorListEvaluateJobs);
const mockUseListExperiments = vi.mocked(useListExperiments);
const mockUseModelsListModels = vi.mocked(useModelsListModels);

const renderStatTileRow = (props: Partial<ComponentProps<typeof StatTileRow>> = {}) =>
  render(
    <MemoryRouter>
      <StatTileRow workspace="my-workspace" {...props} />
    </MemoryRouter>
  );

describe('StatTileRow', () => {
  beforeEach(() => {
    mockUseAgentsListAgents.mockReturnValue(queryResult(1));
    mockUseInsightsListInsights.mockReturnValue(queryResult(4));
    mockUseEvaluatorListEvaluateJobs.mockReturnValue(queryResult(120));
    mockUseListExperiments.mockReturnValue(queryResult(1));
    mockUseModelsListModels.mockReturnValue(queryResult(0));
  });

  it('renders every tile label with its total count', () => {
    renderStatTileRow();

    expect(screen.getByText('Agents')).toBeInTheDocument();
    expect(screen.getAllByText('1')).toHaveLength(2); // Agents and Experiments both show 1
    expect(screen.getByText('Insights')).toBeInTheDocument();
    expect(screen.getByText('4')).toBeInTheDocument();
    expect(screen.getByText('Test Cases')).toBeInTheDocument();
    expect(screen.getByText('120')).toBeInTheDocument();
    expect(screen.getByText('Experiments')).toBeInTheDocument();
    expect(screen.getByText('Custom Models')).toBeInTheDocument();
    expect(screen.getByText('0')).toBeInTheDocument();
  });

  it('links each tile to its section route', () => {
    renderStatTileRow();

    expect(screen.getByRole('link', { name: /Agents/ })).toHaveAttribute(
      'href',
      '/workspaces/my-workspace/agents'
    );
    expect(screen.getByRole('link', { name: /Experiments/ })).toHaveAttribute(
      'href',
      '/workspaces/my-workspace/experiment'
    );
  });

  it('shows a loading placeholder instead of a stale count', () => {
    mockUseAgentsListAgents.mockReturnValue(queryResult(undefined, { isLoading: true }));
    renderStatTileRow();

    expect(screen.getByText('—')).toBeInTheDocument();
  });

  it('shows a placeholder (not a false zero) and an error status when a fetch fails', () => {
    mockUseAgentsListAgents.mockReturnValue(queryResult(undefined, { isError: true }));
    renderStatTileRow();

    expect(screen.getByText('—')).toBeInTheDocument();
    // Custom Models is mocked to a genuine 0 above — only that tile should show it.
    expect(screen.getAllByText('0')).toHaveLength(1);
  });

  it('hides a tile whose backing feature flag is disabled', () => {
    renderStatTileRow({ experimentEnabled: false });

    expect(screen.queryByText('Experiments')).not.toBeInTheDocument();
    expect(screen.getByText('Agents')).toBeInTheDocument();
  });

  it('renders nothing when every backing flag is disabled', () => {
    const { container } = renderStatTileRow({
      agentsEnabled: false,
      optimizerEnabled: false,
      evaluatorEnabled: false,
      experimentEnabled: false,
      customizerEnabled: false,
    });

    expect(container).toBeEmptyDOMElement();
  });

  it('scopes the Custom Models count to models with a base_model or adapters', () => {
    renderStatTileRow();

    const params = mockUseModelsListModels.mock.calls[0]![1]!;
    const filter = JSON.parse(params.filter as unknown as string);
    expect(filter).toEqual({
      $or: [{ 'data.base_model': { $not: { $eq: null } } }, { adapters: { $exists: true } }],
    });
  });

  it('sizes tiles to fill the row width regardless of screen size or tile count', () => {
    renderStatTileRow();

    const grid = screen.getByTestId('nv-grid');
    expect(grid.style.getPropertyValue('--nv-grid-template-columns')).toBe(
      'repeat(auto-fit, minmax(200px, 1fr))'
    );
  });
});
