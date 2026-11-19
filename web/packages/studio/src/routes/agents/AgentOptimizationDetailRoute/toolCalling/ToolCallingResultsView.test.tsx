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
import {
  METRIC_DESCRIPTIONS,
  SIGNAL_DESCRIPTIONS,
  SIGNAL_ROWS,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/toolCalling/toolCallingResults';
import { getAgentOptimizationDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

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
    expect(
      screen.getByText('Each column runs the same prompts; each fix builds on the one before it.')
    ).toBeInTheDocument();
    for (const label of ['Current agent', 'Prompt fix', 'Schema fix']) {
      expect(screen.getAllByRole('columnheader', { name: new RegExp(`^${label} `) }).length).toBe(
        3
      );
    }
    expect(screen.getByRole('img', { name: 'About Tool error rate' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: '80%' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: '6% −74 pts' })).toBeInTheDocument();
    expect(screen.queryByText('80% → 6%')).not.toBeInTheDocument();
    expect(screen.queryByTestId('stat-tile-surface')).not.toBeInTheDocument();
  });

  it('marks the recommended variant column and its suggestion, and nothing else', async () => {
    renderRun();

    expect(
      await screen.findByRole('columnheader', { name: /^Prompt fix .*Recommended/ })
    ).toBeInTheDocument();
    expect(screen.getAllByText('Recommended')).toHaveLength(2);
    expect(
      within(screen.getByTestId('suggestion-config-fix')).getByText('Recommended')
    ).toBeInTheDocument();
    expect(
      within(screen.getByTestId('suggestion-schema-fix')).queryByText('Recommended')
    ).not.toBeInTheDocument();
  });

  it.each([
    ['null', { ...TOOL_CALLING_RESULT_FIXTURE, recommended_variant: null }],
    ['missing', { ...TOOL_CALLING_RESULT_FIXTURE, recommended_variant: undefined }],
  ])('marks nothing when the recommended variant is %s', async (_case, result) => {
    renderRun({ resultBody: JSON.stringify(result) });

    expect(await screen.findByTestId('suggestion-config-fix')).toBeInTheDocument();
    expect(screen.queryByText('Recommended')).not.toBeInTheDocument();
  });

  it('breaks down how bad calls were detected per variant, marking error text as heuristic', async () => {
    renderRun();

    expect(await screen.findByText('How bad calls were detected')).toBeInTheDocument();
    for (const row of SIGNAL_ROWS) {
      expect(screen.getByRole('img', { name: `About ${row.label}` })).toBeInTheDocument();
    }
    expect(screen.getByText('Error text (heuristic)')).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: '40' })).toBeInTheDocument();
  });

  it.each([
    ['a metric', 'About Tool error rate', METRIC_DESCRIPTIONS.tool_error_rate],
    ['a signal', 'About Schema invalid', SIGNAL_DESCRIPTIONS.schema_invalid],
    [
      'the recommended badge',
      'About Recommended',
      'The variant the recommendation above tells you to adopt.',
    ],
    [
      'a suggestion kind',
      'About Tool schema',
      "Changes the tool's input schema. Send it to the tool's owner.",
    ],
  ])('explains %s in a tooltip on hover', async (_subject, trigger, content) => {
    renderRun();

    const [icon] = await screen.findAllByRole('img', { name: trigger });
    await userEvent.hover(icon);
    await waitFor(() => expect(icon).toHaveAccessibleDescription(content));
  });

  it('explains each variant from its own description and skips the icon when there is none', async () => {
    const [baseline, configFix, schemaFix] = TOOL_CALLING_RESULT_FIXTURE.variants;
    renderRun({
      resultBody: JSON.stringify({
        ...TOOL_CALLING_RESULT_FIXTURE,
        variants: [
          { ...baseline, label: 'As shipped' },
          { ...configFix, label: 'Tweaked prompt', description: 'Prompt changed by the run.' },
          { ...schemaFix, description: '' },
        ],
      }),
    });

    const [icon] = await screen.findAllByRole('img', { name: 'About Tweaked prompt' });
    expect(screen.getAllByRole('columnheader', { name: /^As shipped/ })).toHaveLength(3);
    expect(
      screen.queryByRole('img', { name: `About ${schemaFix?.label}` })
    ).not.toBeInTheDocument();
    await userEvent.hover(icon);
    await waitFor(() => expect(icon).toHaveAccessibleDescription('Prompt changed by the run.'));
  });

  it('shows signal chips and schema errors on a rejected call', async () => {
    renderRun();

    const cells = within(await screen.findByRole('row', { name: /^product-reviews/ }));
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
      within(configFix).getByText('Verified by Prompt fix · Tool error rate 80% → 47%')
    ).toBeInTheDocument();
    const diff = await within(configFix).findByLabelText('candidate-agent.diff');
    expect(
      within(diff).getByText(/system_prompt: Build a Data Designer job with model/)
    ).toHaveAttribute('data-diff-line', 'added');
    expect(within(diff).getByText(/system_prompt: Build a Data Designer job\.$/)).toHaveAttribute(
      'data-diff-line',
      'removed'
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
