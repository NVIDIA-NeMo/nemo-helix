// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import {
  TOOL_CALLING_RESULT_FIXTURE,
  toolCallingRunHandlers,
} from '@studio/mocks/handlers/toolCallingRun';
import { server } from '@studio/mocks/node';
import { AgentOptimizationDetailRoute } from '@studio/routes/agents/AgentOptimizationDetailRoute';
import { getAgentOptimizationDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen, within } from '@studio/tests/util/render';

const workspace = workspace1.workspace;
const jobName = 'fix-tool-calls';

const renderRun = (options: Partial<Parameters<typeof toolCallingRunHandlers>[0]> = {}) => {
  server.use(...toolCallingRunHandlers({ workspace, jobName, ...options }));
  return renderRoute(undefined, {
    history: getAgentOptimizationDetailRoute(workspace, jobName),
    routes: [
      { path: ROUTES.workspace.agentOptimizationDetail, element: <AgentOptimizationDetailRoute /> },
    ],
  });
};

describe('AgentOptimizationDetailRoute — tool-calling', () => {
  it('leads with the recommendation', async () => {
    renderRun();

    expect(
      await screen.findByText(TOOL_CALLING_RESULT_FIXTURE.recommendation ?? '')
    ).toBeInTheDocument();
    expect(
      screen.queryByText('This study did not register any trial results.')
    ).not.toBeInTheDocument();
  });

  it('shows every variant before and after, with deltas against the baseline', async () => {
    renderRun();

    expect(await screen.findByText('Before and after')).toBeInTheDocument();
    expect(screen.getAllByRole('columnheader', { name: 'Current agent' }).length).toBeGreaterThan(
      0
    );
    expect(screen.getAllByRole('columnheader', { name: 'Config fix' }).length).toBeGreaterThan(0);
    expect(
      screen.getAllByRole('columnheader', { name: 'Config + schema fix' }).length
    ).toBeGreaterThan(0);
    expect(screen.getByRole('cell', { name: 'Tool error rate' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: '80%' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: '6% −74 pts' })).toBeInTheDocument();
    expect(screen.getByText('80% → 6%')).toBeInTheDocument();
  });

  it('breaks down how bad calls were detected per variant, marking error text as heuristic', async () => {
    renderRun();

    expect(await screen.findByText('How bad calls were detected')).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: 'Schema invalid' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: 'Error text (heuristic)' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: 'Runs with a retry loop' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: '40' })).toBeInTheDocument();
  });

  it('shows signal chips and schema errors on a rejected call', async () => {
    renderRun();

    const row = (await screen.findByText('product-reviews')).closest('tr');
    expect(row).not.toBeNull();
    const cells = within(row as HTMLElement);
    expect(cells.getByText('Tool error')).toBeInTheDocument();
    expect(cells.getByText('Schema invalid')).toBeInTheDocument();
    expect(cells.getByText(/'model_configs' was unexpected/)).toBeInTheDocument();
  });

  it('reads a result written before detection signals existed', async () => {
    const legacy = {
      ...TOOL_CALLING_RESULT_FIXTURE,
      variants: TOOL_CALLING_RESULT_FIXTURE.variants.map(({ metrics, examples, ...variant }) => ({
        ...variant,
        metrics: { ...metrics, signals: undefined },
        examples: examples.map((example) => ({
          ...example,
          signals: undefined,
          schema_errors: undefined,
        })),
      })),
    };
    renderRun({ resultBody: JSON.stringify(legacy) });

    expect(await screen.findByText('Before and after')).toBeInTheDocument();
    expect(screen.queryByText('How bad calls were detected')).not.toBeInTheDocument();
    expect(screen.getByText('product-reviews')).toBeInTheDocument();
  });

  it('lists rejection reasons, a rejected call, and the findings', async () => {
    renderRun();

    expect(await screen.findByText('corrupted or unknown key')).toBeInTheDocument();
    expect(screen.getByText('model/provider not on platform')).toBeInTheDocument();
    expect(screen.getByText('product-reviews')).toBeInTheDocument();
    expect(screen.getByText("columns[0]: unknown field 'sampler_type '")).toBeInTheDocument();
    expect(screen.getByText('Column items allow any key')).toBeInTheDocument();
    expect(
      screen.getByText('Schema lint · generate_data_designer_job_request')
    ).toBeInTheDocument();
    expect(screen.getByText('Failure analysis')).toBeInTheDocument();
  });

  it('shows each suggestion with who applies it, its verified delta, and its diff', async () => {
    renderRun();

    const configFix = await screen.findByTestId('suggestion-config-fix');
    expect(within(configFix).getByText('Applies automatically')).toBeInTheDocument();
    expect(
      within(configFix).getByText('Verified by Config fix · Tool error rate 80% → 47%')
    ).toBeInTheDocument();
    const diff = await within(configFix).findByLabelText('candidate-agent.diff');
    expect(diff.querySelector('[data-diff-line="added"]')).toHaveTextContent(
      'system_prompt: Build a Data Designer job with model default/nemotron-3-5-super.'
    );
    expect(diff.querySelector('[data-diff-line="removed"]')).toHaveTextContent(
      'system_prompt: Build a Data Designer job.'
    );
    expect(
      within(configFix).getByRole('button', { name: /Download candidate-agent.yaml/ })
    ).toBeInTheDocument();

    const schemaFix = screen.getByTestId('suggestion-schema-fix');
    expect(within(schemaFix).getByText("Send to the tool's owner")).toBeInTheDocument();
    expect(await within(schemaFix).findByLabelText('schema-patch.diff')).toHaveTextContent(
      '"additionalProperties": false'
    );
  });

  it('explains when the run wrote no result file', async () => {
    renderRun({ resultBody: null });

    expect(await screen.findByText('No tool-calling results')).toBeInTheDocument();
    expect(screen.getByText('Artifacts')).toBeInTheDocument();
  });

  it('explains when the result file does not match the contract', async () => {
    renderRun({ resultBody: JSON.stringify({ version: 1, agent: 'a', variants: [] }) });

    expect(await screen.findByText('Could not load tool-calling results')).toBeInTheDocument();
    expect(screen.getByText(/does not match the expected shape at variants/)).toBeInTheDocument();
  });

  it('leaves other strategies on the trial reader', async () => {
    renderRun({ strategy: 'config-tuning' });

    expect(
      await screen.findByText('This study did not register any trial results.')
    ).toBeInTheDocument();
    expect(screen.queryByText('Before and after')).not.toBeInTheDocument();
  });
});
