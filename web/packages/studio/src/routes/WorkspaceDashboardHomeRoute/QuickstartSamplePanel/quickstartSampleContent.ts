// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { AGENT_OPTIMIZATIONS_ENABLED, INTAKE_ENABLED } from '@studio/constants/environment';
import {
  getAgentDetailRoute,
  getAgentEvaluationsTabRoute,
  getAgentOptimizationsTabRoute,
  getAgentOptimizeRoute,
  getAgentRunEvaluationRoute,
  getIntakeTracesRoute,
} from '@studio/routes/utils';
import { MessagesSquare, type LucideIcon } from 'lucide-react';

export type QuickstartSampleView = 'studio' | 'cli';

export interface QuickstartSampleAgent {
  readonly name: string;
  readonly description: string;
  /** Looked up in `badgeStatus` (lowercased); anything unmapped renders as "Unknown". */
  readonly status: string;
  /** Overrides the badge text, for states with no status of their own (e.g. no deployments). */
  readonly statusLabel?: string;
  /**
   * Not the agent name: absent an explicit `--name`, deployments are `${agent}-${8 hex}`.
   * Omitted leaves a placeholder in the command rather than a name that would 404.
   */
  readonly deploymentName?: string;
}

export interface QuickstartSampleAction {
  readonly label: string;
  /** Modals on the agent page are reached through its `?action=` param, so this is always a route. */
  readonly href: string;
}

interface QuickstartSampleStepCopy {
  readonly title: string;
  readonly description: string;
}

export interface QuickstartSampleStep {
  readonly id: string;
  readonly icon: LucideIcon;
  readonly studio: QuickstartSampleStepCopy & {
    readonly actions: readonly QuickstartSampleAction[];
  };
  /** Carries its own copy: step 1 is "Chat with the agent" here, "Try the agent" in Studio. */
  readonly cli: QuickstartSampleStepCopy & { readonly commands: readonly string[] };
}

/**
 * These names are unconstrained strings and the snippet goes straight to the clipboard,
 * so an unquoted space, `;` or backtick would word-split or execute.
 */
const shellQuote = (value: string): string => `'${value.replaceAll("'", `'\\''`)}'`;

/**
 * A step whose destination is not registered is dropped outright rather than rendered as a
 * dead link. Both views drop together: the two carry the same steps in the same order, and a
 * step that Studio cannot reach is not one to hand someone a CLI command for either.
 */
export interface QuickstartSampleFeatures {
  /** `getIntakeTracesRoute` — the whole intake group is registered behind this. */
  readonly intakeEnabled?: boolean;
  /** Gates the agent page's Optimizations tab, which is where step 4 points. */
  readonly agentOptimizationsEnabled?: boolean;
}

export interface BuildQuickstartSampleStepsOptions extends QuickstartSampleFeatures {
  readonly workspace: string;
  readonly agent: QuickstartSampleAgent;
}

/**
 * Fixed order, never derived: there is no per-user progress and no completion state.
 * `--workspace` is explicit on every command because the CLI otherwise falls back to `default`,
 * and the point of this panel is the sandbox workspace the reader is currently in.
 *
 * The evaluate step is deliberately ungated: its "View results" destination is the agent page's
 * Evaluations tab, which `AgentDetailRoute` renders unconditionally — `EVALUATOR_ENABLED` gates
 * the standalone evaluator routes this panel never links to.
 */
export const buildQuickstartSampleSteps = ({
  workspace,
  agent,
  intakeEnabled = INTAKE_ENABLED,
  agentOptimizationsEnabled = AGENT_OPTIMIZATIONS_ENABLED,
}: BuildQuickstartSampleStepsOptions): readonly QuickstartSampleStep[] => {
  const ws = shellQuote(workspace);
  const deployment = agent.deploymentName
    ? shellQuote(agent.deploymentName)
    : "'<agent-deployment>'";

  return [
    {
      id: 'try-the-agent',
      icon: MessagesSquare,
      studio: {
        title: 'Try the agent',
        description: 'Chat with the sample agent to see how it responds.',
        actions: [
          {
            label: 'Open and Chat',
            href: `${getAgentDetailRoute(workspace, agent.name)}?tab=chat`,
          },
        ],
      },
      cli: {
        title: 'Chat with the agent',
        description: 'Chat with the sample agent to see how it responds.',
        commands: [
          // Single quotes, not double: interactive zsh (the macOS default) and bash 3.2 read
          // `!"` as history expansion, so a double-quoted "Hello agent!" never runs on paste.
          `nemo agents chat --agent-deployment ${deployment} --input 'Hello agent!' --workspace ${ws}`,
        ],
      },
    },
    ...(intakeEnabled
      ? [
          {
            id: 'inspect-a-trace',
            icon: ENTITY_ICONS.telemetryTraces,
            studio: {
              title: 'Inspect a trace',
              description: 'Inspect sample traces including misclassifications',
              actions: [{ label: 'View Traces', href: getIntakeTracesRoute(workspace) }],
            },
            cli: {
              title: 'Inspect a trace',
              description: 'Inspect sample traces including misclassifications',
              commands: [
                `nemo intake traces list --workspace ${ws}`,
                // Quoted so the copied blob still parses: a bare <TRACE_ID> reads as a
                // shell redirection and dies with a syntax error on paste.
                `nemo intake traces get '<TRACE_ID>' --workspace ${ws}`,
              ],
            },
          },
        ]
      : []),
    {
      id: 'evaluate-the-agent',
      icon: ENTITY_ICONS.evaluationResults,
      studio: {
        title: 'Evaluate the agent',
        description: 'Run your own evaluation or view results from a sample evaluation run.',
        actions: [
          { label: 'View results', href: getAgentEvaluationsTabRoute(workspace, agent.name) },
          // The modal lives on the agent page, which opens it from `?action=` on arrival.
          { label: 'Run Evaluation', href: getAgentRunEvaluationRoute(workspace, agent.name) },
        ],
      },
      cli: {
        title: 'Evaluate the agent',
        description: 'Run your own evaluation or view results from a sample evaluation run.',
        // There is no `submit` subcommand; nemo-evaluator has a regression test
        // asserting that form never ships again. Matches EntityEmptyState/registry.ts.
        commands: [`nemo evaluator evaluate --spec-file '<spec>.json' --workspace ${ws}`],
      },
    },
    ...(agentOptimizationsEnabled
      ? [
          {
            id: 'optimize',
            icon: ENTITY_ICONS.agentOptimizations,
            studio: {
              title: 'Optimize',
              description: 'Run your own optimization or view the sample study.',
              actions: [
                {
                  label: 'View results',
                  href: getAgentOptimizationsTabRoute(workspace, agent.name),
                },
                { label: 'Optimize', href: getAgentOptimizeRoute(workspace, agent.name) },
              ],
            },
            cli: {
              title: 'Optimize',
              description: 'Run your own optimization or view the sample study.',
              // Drives a coding-agent loop on the user's checkout, so `--agent` and
              // `--evals` are required local paths — but the run is still a platform
              // submission, so it is scoped by `--workspace` like the rest.
              commands: [
                `nemo agents optimize-skills --agent '<agent-dir>' --evals '<evals-dir>' --workspace ${ws}`,
              ],
            },
          },
        ]
      : []),
  ];
};
