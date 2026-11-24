// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { agentsCreateExecuteJob, agentsRegisterAgent } from '@nemo/sdk/generated/agents/agents';
import { agentsListPluginManagedAgentTypesRoute } from '@nemo/sdk/generated/agents/plugin-managed-agent-types';
import type { ExecuteJob } from '@nemo/sdk/generated/agents/schema';
import { compassGetAnalysisConfig } from '@nemo/sdk/generated/compass/compass-analysis-configs';
import type { AnalysisConfig } from '@nemo/sdk/generated/compass/schema';
import type { AtifIngestRequest } from '@nemo/sdk/generated/platform/schema';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema/HelixJobStatus';
import { AxiosError } from 'axios';

export const COMPASS_AGENT_TYPE = 'compass.analyst';

export type InsightsTriggerStatus = 'started' | 'not-enabled' | 'error';

export interface InsightsTriggerResult {
  agent: string;
  status: InsightsTriggerStatus;
  jobName?: string;
  message?: string;
}

/**
 * The agents named by a set of imported trajectories, deduplicated and in first-seen
 * order. Intake scopes analysis by `agent_name`, which ATIF derives from `agent.name`.
 */
export const agentsFromTrajectories = (trajectories: AtifIngestRequest[]): string[] => [
  ...new Set(trajectories.map(({ agent }) => agent?.name).filter((name): name is string => !!name)),
];

/**
 * A default/fast pair entered in the modal, replacing whatever the AnalysisConfig
 * stored. Either half may be left blank to keep the stored value for that half.
 */
export interface InsightsModelOverrides {
  default_model?: string;
  fast_model?: string;
}

export interface AnalysisRunOptions {
  /** Analyze only traces from this time on (ISO 8601). */
  since?: string;
  /** Analyze only the spans of this evaluation; takes an evaluation's name. */
  evaluation_id?: string;
  /** Skips the config lookup when the caller already holds the agent's config. */
  storedConfig?: AnalysisConfig;
  /** Resolves the Compass agent; lets a batch of runs resolve it once. */
  resolveAgent?: () => Promise<string>;
}

/**
 * Studio stores Model Entity references in `workspace/name` format.
 */
export const isQualifiedModelRef = (ref: string): boolean => {
  const [workspace, ...rest] = ref.split('/');
  return rest.length === 1 && workspace.length > 0 && rest[0].length > 0;
};

const isHelixJobStatus = (value: unknown): value is HelixJobStatus =>
  Object.values<unknown>(HelixJobStatus).includes(value);

export const analysisJobStatus = (
  job?: Pick<ExecuteJob, 'status'> | null
): HelixJobStatus | undefined => {
  const status = job?.status;
  return isHelixJobStatus(status) ? status : undefined;
};

const statusOf = (error: unknown): number | undefined =>
  error instanceof AxiosError ? error.response?.status : undefined;

const messageOf = (error: unknown): string => {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (detail && typeof detail === 'object' && typeof detail.error === 'string') {
      return detail.error;
    }
  }
  return error instanceof Error ? error.message : 'Unknown error.';
};

/**
 * Returns the name of the workspace's Compass agent, creating it when the workspace has none.
 *
 * Saving an analysis config creates the agent, so this only creates it for a workspace
 * whose config predates the agent type.
 */
export const resolveCompassAgent = async (workspace: string): Promise<string> => {
  const findType = async () => {
    const types = await agentsListPluginManagedAgentTypesRoute(workspace);
    const compass = types.find(({ type }) => type === COMPASS_AGENT_TYPE);
    if (!compass) {
      throw new Error(`The "${COMPASS_AGENT_TYPE}" agent type is not installed on this platform.`);
    }
    return compass;
  };

  const compass = await findType();
  if (compass.agent) {
    return compass.agent;
  }
  try {
    const created = await agentsRegisterAgent(workspace, {
      name: compass.default_name,
      plugin_managed_agent: { type: COMPASS_AGENT_TYPE },
    });
    return created.name ?? compass.default_name;
  } catch (error) {
    if (statusOf(error) !== 409) {
      throw error;
    }
    // Either another request created the Compass agent first, or an unrelated agent has the name.
    const { agent } = await findType();
    if (agent) {
      return agent;
    }
    throw new Error(
      `An agent named "${compass.default_name}" already exists and is not a Compass agent. ` +
        `Create the Compass agent under another name: nemo agents create ${COMPASS_AGENT_TYPE} --name <name> --workspace ${workspace}`
    );
  }
};

/**
 * Queues one Compass run per agent, as an execute job on the workspace's Compass agent.
 *
 * The analysis run needs the default/fast model pair, which only exists on the
 * agent's AnalysisConfig — so an agent that has never been enabled reports
 * `not-enabled` rather than failing the import it followed.
 */
export const triggerInsightsRun = async (
  workspace: string,
  agent: string,
  overrides: InsightsModelOverrides = {},
  {
    since,
    evaluation_id,
    storedConfig,
    resolveAgent = () => resolveCompassAgent(workspace),
  }: AnalysisRunOptions = {}
): Promise<InsightsTriggerResult> => {
  const invalidOverride = [overrides.default_model, overrides.fast_model]
    .map((ref) => ref?.trim())
    .find((ref): ref is string => !!ref && !isQualifiedModelRef(ref));
  if (invalidOverride) {
    return {
      agent,
      status: 'error',
      message: `Model reference "${invalidOverride}" must use workspace/name format (for example "default/model-name").`,
    };
  }

  let config;
  try {
    config = storedConfig ?? (await compassGetAnalysisConfig(workspace, agent));
  } catch (error) {
    if (statusOf(error) === 404) {
      return {
        agent,
        status: 'not-enabled',
        message: `Insights analysis is not enabled for "${agent}". Run: nemo compass analysis enable --agent ${agent}`,
      };
    }
    return { agent, status: 'error', message: messageOf(error) };
  }

  const defaultModel = overrides.default_model?.trim() || config.default_model;
  const fastModel = overrides.fast_model?.trim() || config.fast_model;

  if (!defaultModel || !fastModel) {
    return {
      agent,
      status: 'not-enabled',
      message: `The analysis config for "${agent}" has no model pair. Re-run: nemo compass analysis enable --agent ${agent}`,
    };
  }

  const unqualified = [defaultModel, fastModel].filter((ref) => !isQualifiedModelRef(ref));
  if (unqualified.length > 0) {
    return {
      agent,
      status: 'error',
      message: `Model reference "${unqualified[0]}" must use workspace/name format (for example "default/model-name").`,
    };
  }

  try {
    const job = await agentsCreateExecuteJob(workspace, {
      spec: {
        agent: await resolveAgent(),
        context: {
          agent,
          default_model: defaultModel,
          fast_model: fastModel,
          ...(since ? { since } : {}),
          ...(evaluation_id ? { evaluation_id } : {}),
        },
      },
    });
    return { agent, status: 'started', jobName: job.name };
  } catch (error) {
    return { agent, status: 'error', message: messageOf(error) };
  }
};

/** Runs the per-agent triggers in sequence so the results read in a stable order. */
export const triggerInsightsRuns = async (
  workspace: string,
  agents: string[],
  overrides: InsightsModelOverrides = {}
): Promise<InsightsTriggerResult[]> => {
  // Resolved on first use, so a batch with no runnable agent never touches the Compass agent.
  let compassAgent: Promise<string> | undefined;
  const resolveAgent = () => (compassAgent ??= resolveCompassAgent(workspace));
  const results: InsightsTriggerResult[] = [];
  for (const agent of agents) {
    results.push(await triggerInsightsRun(workspace, agent, overrides, { resolveAgent }));
  }
  return results;
};
