// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { insightsGetAnalysisConfig } from '@nemo/sdk/generated/insights/insights-analysis-configs';
import { insightsCreateAnalyzeJob } from '@nemo/sdk/generated/insights/insights-analysis-jobs';
import type { AtifIngestRequest } from '@nemo/sdk/generated/platform/schema';
import { AxiosError } from 'axios';

export type InsightsTriggerStatus = 'started' | 'not-enabled' | 'error';

export interface InsightsTriggerResult {
  agent: string;
  status: InsightsTriggerStatus;
  /** Name of the created analyze-job, when one was created. */
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

/**
 * Model Entity IDs are always `workspace/name`. An unqualified ref survives job
 * creation and only fails inside the running job, so it is rejected here instead.
 */
export const isQualifiedModelRef = (ref: string): boolean => {
  const [workspace, ...rest] = ref.split('/');
  return rest.length === 1 && workspace.length > 0 && rest[0].length > 0;
};

const statusOf = (error: unknown): number | undefined =>
  error instanceof AxiosError ? error.response?.status : undefined;

const messageOf = (error: unknown): string => {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === 'string') return detail;
  }
  return error instanceof Error ? error.message : 'Unknown error.';
};

/**
 * Queues one insights analyst run per agent.
 *
 * The analyze-job spec needs the default/fast model pair, which only exists on the
 * agent's AnalysisConfig — so an agent that has never been enabled reports
 * `not-enabled` rather than failing the import it followed.
 */
export const triggerInsightsRun = async (
  workspace: string,
  agent: string,
  overrides: InsightsModelOverrides = {}
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
    config = await insightsGetAnalysisConfig(workspace, agent);
  } catch (error) {
    if (statusOf(error) === 404) {
      return {
        agent,
        status: 'not-enabled',
        message: `Insights analysis is not enabled for "${agent}". Run: nemo insights analysis enable --agent ${agent}`,
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
      message: `The analysis config for "${agent}" has no model pair. Re-run: nemo insights analysis enable --agent ${agent}`,
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
    const job = await insightsCreateAnalyzeJob(workspace, {
      description: `Insights analysis triggered by a trace import for ${agent}.`,
      spec: {
        agent,
        default_model: defaultModel,
        fast_model: fastModel,
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
  const results: InsightsTriggerResult[] = [];
  for (const agent of agents) {
    results.push(await triggerInsightsRun(workspace, agent, overrides));
  }
  return results;
};
