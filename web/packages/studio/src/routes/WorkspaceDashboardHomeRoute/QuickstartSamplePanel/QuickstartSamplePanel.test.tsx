// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import {
  getAgentDetailRoute,
  getAgentEvaluationsTabRoute,
  getAgentOptimizationsTabRoute,
  getAgentOptimizeRoute,
  getAgentRunEvaluationRoute,
  getIntakeTracesRoute,
} from '@studio/routes/utils';
import { QuickstartSamplePanel } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSamplePanel';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ComponentProps } from 'react';
import { MemoryRouter } from 'react-router';

const AGENT_NAME = 'email-security-triage';
const DEPLOYMENT_NAME = 'email-security-triage-9f2a1c';
const AGENT_DESCRIPTION = 'Triages inbound email for phishing and spoofing signals.';
const WS = DEFAULT_WORKSPACE;

/** Hardcoded rather than imported — the test is the spec for this copy and its order. */
const STUDIO_STEP_TITLES = ['Try the agent', 'Inspect a trace', 'Evaluate the agent', 'Optimize'];
const CLI_STEP_TITLES = [
  'Chat with the agent',
  'Inspect a trace',
  'Evaluate the agent',
  'Optimize',
];

const renderPanel = (props: Partial<ComponentProps<typeof QuickstartSamplePanel>> = {}) =>
  render(
    <MemoryRouter>
      <QuickstartSamplePanel
        workspace={WS}
        agent={{
          name: AGENT_NAME,
          description: AGENT_DESCRIPTION,
          status: 'Running',
          deploymentName: DEPLOYMENT_NAME,
        }}
        {...props}
      />
    </MemoryRouter>
  );

/** Steps are list items, so they are addressable positionally without `.closest()`. */
const stepAt = (index: number) => screen.getAllByRole('listitem')[index];

const stepTitles = () =>
  screen.getAllByRole('listitem').map((step) => within(step).getByRole('heading').textContent);

/**
 * All four snippets highlight asynchronously; awaiting only one lets a later `setState`
 * land outside `act()`, where `vitest-fail-on-console` blames an unrelated test.
 */
const waitForAllSnippets = async () => {
  await waitFor(() =>
    expect(screen.getAllByTestId('nv-code-snippet-code')).toHaveLength(STUDIO_STEP_TITLES.length)
  );
};

describe('QuickstartSamplePanel', () => {
  describe('the sandboxed agent', () => {
    it('shows the agent name, description, and status', () => {
      renderPanel();

      const row = screen.getByTestId('quickstart-sample-agent-row');
      expect(within(row).getByText(AGENT_NAME)).toBeInTheDocument();
      expect(within(row).getByText(AGENT_DESCRIPTION)).toBeInTheDocument();
      expect(within(row).getByText('Running')).toBeInTheDocument();
    });

    it('reflects a non-running status rather than hardcoding "Running"', () => {
      renderPanel({
        agent: { name: AGENT_NAME, description: AGENT_DESCRIPTION, status: 'Failed' },
      });

      expect(screen.getByText('Failed')).toBeInTheDocument();
      expect(screen.queryByText('Running')).not.toBeInTheDocument();
    });

    it('links the whole row to the agent page', () => {
      renderPanel();

      expect(screen.getByTestId('quickstart-sample-agent-row')).toHaveAttribute(
        'href',
        getAgentDetailRoute(WS, AGENT_NAME)
      );
    });

    it('truncates a long description to a single line', () => {
      // happy-dom has no layout engine, so the class is the only assertable signal.
      // The visual proof lives in the `LongAgentDescription` story.
      renderPanel({
        agent: { name: AGENT_NAME, description: 'x'.repeat(400), status: 'Running' },
      });

      expect(screen.getByTestId('quickstart-sample-agent-description')).toHaveClass('truncate');
    });

    it('renders nothing when no sample agent is installed', () => {
      const { container } = renderPanel({ agent: undefined });

      expect(container).toBeEmptyDOMElement();
    });
  });

  describe('the steps', () => {
    it('renders exactly four steps in a fixed order', () => {
      renderPanel();

      expect(stepTitles()).toEqual(STUDIO_STEP_TITLES);
    });

    it('keeps the same steps in the same order in the CLI view', async () => {
      renderPanel({ defaultView: 'cli' });
      await waitForAllSnippets();

      expect(stepTitles()).toEqual(CLI_STEP_TITLES);
    });

    it('exposes the steps as a list even where CSS strips list semantics', () => {
      renderPanel();

      expect(screen.getByRole('list')).toBeInTheDocument();
    });

    it('closes the timeline by dropping the connector after the last step', () => {
      renderPanel();

      expect(screen.getAllByTestId('quickstart-sample-step-connector')).toHaveLength(
        STUDIO_STEP_TITLES.length - 1
      );
    });

    it('gives a single-action step only its own action', () => {
      renderPanel();

      expect(within(stepAt(0)).getByRole('link', { name: 'Open and Chat' })).toBeInTheDocument();
      expect(within(stepAt(0)).getAllByRole('link')).toHaveLength(1);

      expect(within(stepAt(1)).getByRole('link', { name: 'View Traces' })).toBeInTheDocument();
      expect(within(stepAt(1)).getAllByRole('link')).toHaveLength(1);
    });

    it('deep-links each action to its own destination', () => {
      renderPanel();

      expect(within(stepAt(0)).getByRole('link', { name: 'Open and Chat' })).toHaveAttribute(
        'href',
        `${getAgentDetailRoute(WS, AGENT_NAME)}?tab=chat`
      );
      expect(within(stepAt(1)).getByRole('link', { name: 'View Traces' })).toHaveAttribute(
        'href',
        getIntakeTracesRoute(WS)
      );
      expect(within(stepAt(2)).getByRole('link', { name: 'View results' })).toHaveAttribute(
        'href',
        getAgentEvaluationsTabRoute(WS, AGENT_NAME)
      );
      expect(within(stepAt(3)).getByRole('link', { name: 'View results' })).toHaveAttribute(
        'href',
        getAgentOptimizationsTabRoute(WS, AGENT_NAME)
      );
    });

    it('links the modal actions to the agent page, which opens them on arrival', () => {
      renderPanel();

      // Links, not buttons — the modals live on the agent page. Scoped to the step, since
      // step 4's heading is also "Optimize".
      expect(within(stepAt(2)).getByRole('link', { name: 'Run Evaluation' })).toHaveAttribute(
        'href',
        getAgentRunEvaluationRoute(WS, AGENT_NAME)
      );
      expect(within(stepAt(3)).getByRole('link', { name: 'Optimize' })).toHaveAttribute(
        'href',
        getAgentOptimizeRoute(WS, AGENT_NAME)
      );
    });
  });

  describe('feature flags', () => {
    it('drops the trace step when the intake routes are not registered', () => {
      renderPanel({ intakeEnabled: false });

      // Dropped outright, not left as a link into a route group that does not exist.
      expect(stepTitles()).toEqual(['Try the agent', 'Evaluate the agent', 'Optimize']);
      expect(screen.queryByRole('link', { name: 'View Traces' })).not.toBeInTheDocument();
    });

    it('drops the optimize step when the Optimizations tab is disabled', () => {
      renderPanel({ agentOptimizationsEnabled: false });

      // `isAgentDetailTab` rejects `?tab=optimizations` under this flag, so the link
      // would land on Overview rather than anywhere it claims to go.
      expect(stepTitles()).toEqual(['Try the agent', 'Inspect a trace', 'Evaluate the agent']);
      expect(
        screen.queryByRole('link', { name: 'View results' })?.getAttribute('href')
      ).not.toContain('tab=optimizations');
    });

    it('keeps the evaluate step, whose tab is not behind a flag', () => {
      renderPanel({ intakeEnabled: false, agentOptimizationsEnabled: false });

      expect(stepTitles()).toContain('Evaluate the agent');
    });

    it('drops the connector for whatever step ends up last', () => {
      renderPanel({ agentOptimizationsEnabled: false });

      expect(screen.getAllByTestId('quickstart-sample-step-connector')).toHaveLength(2);
    });

    it('renders nothing when the agents routes are disabled', () => {
      const { container } = renderPanel({ agentsEnabled: false });

      expect(container).toBeEmptyDOMElement();
    });

    it('renders nothing rather than empty chrome when every step is gated out', () => {
      const { container } = renderPanel({ steps: [] });

      expect(container).toBeEmptyDOMElement();
    });
  });

  describe('the Studio / CLI switch', () => {
    it('is labelled and starts on NeMo Studio', () => {
      renderPanel();

      expect(screen.getByRole('radiogroup', { name: 'Quickstart view' })).toBeInTheDocument();
      expect(screen.getByRole('radio', { name: 'NeMo Studio' })).toBeChecked();
      expect(screen.getByRole('radio', { name: 'NeMo CLI' })).not.toBeChecked();
    });

    it('swaps every action button for a nemo command when NeMo CLI is selected', async () => {
      const user = userEvent.setup();
      renderPanel();

      await user.click(screen.getByRole('radio', { name: 'NeMo CLI' }));
      await waitForAllSnippets();

      expect(screen.queryByRole('link', { name: 'Open and Chat' })).not.toBeInTheDocument();
      expect(screen.queryByRole('link', { name: 'View Traces' })).not.toBeInTheDocument();
    });

    it('targets the deployment and the workspace, not the agent and the default', async () => {
      renderPanel({ defaultView: 'cli' });
      await waitForAllSnippets();

      // --agent-deployment takes an AgentDeployment name, which is not the agent name.
      expect(stepAt(0)).toHaveTextContent(`--agent-deployment '${DEPLOYMENT_NAME}'`);

      // Every command, not just the first two: the CLI falls back to the `default`
      // workspace, and the whole point of this panel is the sandbox the reader is in.
      STUDIO_STEP_TITLES.forEach((_title, index) => {
        expect(stepAt(index)).toHaveTextContent(`--workspace '${WS}'`);
      });
    });

    it('leaves a placeholder when the deployment name is unknown', async () => {
      renderPanel({
        defaultView: 'cli',
        agent: { name: AGENT_NAME, description: AGENT_DESCRIPTION, status: 'Running' },
      });
      await waitForAllSnippets();

      expect(stepAt(0)).toHaveTextContent("--agent-deployment '<agent-deployment>'");
      expect(stepAt(0)).not.toHaveTextContent(`--agent-deployment '${AGENT_NAME}'`);
    });

    it('uses the real evaluator command, not the retired `evaluate submit` form', async () => {
      renderPanel({ defaultView: 'cli' });
      await waitForAllSnippets();

      expect(within(stepAt(2)).getByTestId('nv-code-snippet-code').textContent).toBe(
        ['nemo evals evaluate', "--spec-file '<spec>.json'", `--workspace '${WS}'`].join(' \\\n  ')
      );
      expect(stepAt(2)).not.toHaveTextContent('evaluate submit');
    });

    it('optimizes with the same run-strategy study Studio submits', async () => {
      renderPanel({ defaultView: 'cli' });
      await waitForAllSnippets();

      // `optimize` alone is a command group, and `optimize-skills` jobs never reach the
      // Optimizations tab that "View results" opens. Exact text, so the one-flag-per-line
      // `\` continuations are checked too — a stray character after one breaks the paste.
      expect(within(stepAt(3)).getByTestId('nv-code-snippet-code').textContent).toBe(
        [
          [
            'nemo agents optimize prepare-fileset',
            "--source '<bundle-dir>'",
            "--optimize-config 'optimize.yaml'",
            `--fileset '${AGENT_NAME}-optimize'`,
            `--agent '${AGENT_NAME}'`,
            `--workspace '${WS}'`,
          ].join(' \\\n  '),
          [
            'nemo agents optimize run-strategy',
            '--strategy legacy',
            `--agent '${AGENT_NAME}'`,
            `--optimize-config-fileset '${WS}/${AGENT_NAME}-optimize'`,
            "--optimize-config 'optimize.yaml'",
            `--workspace '${WS}'`,
          ].join(' \\\n  '),
        ].join('\n\n')
      );
    });

    it('shows both commands for a step that needs two', async () => {
      renderPanel({ defaultView: 'cli' });
      await waitForAllSnippets();

      // Separated by a blank line, so the two read as separate commands.
      expect(within(stepAt(1)).getByTestId('nv-code-snippet-code').textContent).toBe(
        `nemo intake traces list --workspace '${WS}'\n\nnemo intake traces get '<TRACE_ID>' --workspace '${WS}'`
      );
    });

    it('prefixes every command with a shell prompt', async () => {
      renderPanel({ defaultView: 'cli' });
      await waitForAllSnippets();

      expect(screen.getAllByText('>_')).toHaveLength(STUDIO_STEP_TITLES.length);
    });

    it('shows no shell prompt in the Studio view', () => {
      renderPanel();

      expect(screen.queryByText('>_')).not.toBeInTheDocument();
    });

    it('quotes placeholders so a copied command still parses in a shell', async () => {
      renderPanel({ defaultView: 'cli' });
      await waitForAllSnippets();

      // Bare <TRACE_ID> reads as a redirection and makes the pasted blob a syntax error.
      expect(stepAt(1)).toHaveTextContent("'<TRACE_ID>'");
    });

    it('reports the selected view to the caller', async () => {
      const onViewChange = vi.fn();
      const user = userEvent.setup();
      renderPanel({ onViewChange });

      await user.click(screen.getByRole('radio', { name: 'NeMo CLI' }));
      await waitForAllSnippets();

      expect(onViewChange).toHaveBeenCalledWith('cli');
    });

    it("copies that step's command to the clipboard", async () => {
      // `userEvent.setup()` stubs `navigator.clipboard`; without it KUI's copy button
      // console.errors on failure and `vitest-fail-on-console` fails the test.
      const user = userEvent.setup();
      renderPanel({ defaultView: 'cli' });
      await waitForAllSnippets();

      await user.click(
        within(stepAt(2)).getByRole('button', { name: 'Copy the Evaluate the agent command' })
      );

      expect(await navigator.clipboard.readText()).toContain('nemo evals evaluate');
    });
  });

  describe('the footer', () => {
    it('offers the shared-workspace hand-off', async () => {
      const onSwitchWorkspace = vi.fn();
      const user = userEvent.setup();
      renderPanel({ onSwitchWorkspace });

      expect(screen.getByText('Ready to start with your own assets?')).toBeInTheDocument();
      await user.click(screen.getByRole('button', { name: 'Switch to Shared Workspace' }));

      expect(onSwitchWorkspace).toHaveBeenCalledTimes(1);
    });

    it('is hidden when the caller cannot act on it', () => {
      renderPanel();

      expect(screen.queryByText('Ready to start with your own assets?')).not.toBeInTheDocument();
      expect(
        screen.queryByRole('button', { name: 'Switch to Shared Workspace' })
      ).not.toBeInTheDocument();
    });
  });
});
