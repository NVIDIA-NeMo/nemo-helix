// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isConflictError } from '@nemo/common/src/api/common/utils';
import { toValidEntityName } from '@nemo/common/src/utils/entityName';
import { createExperiment, getExperiment } from '@nemo/sdk/generated/platform/experiments';
import type { ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { bareName } from '@studio/components/evaluation/submitEvaluationJob';

/** Experiment metadata key marking the per-agent catch-all for runs started without an
 *  experiment. Intake requires every Evaluation to belong to at least one experiment, so an
 *  unlinked run lands here until it is linked to a real one. */
export const DEFAULT_EXPERIMENT_AGENT_KEY = 'studio_default_for_agent';

export const defaultExperimentName = (agent: string): string =>
  toValidEntityName(`${bareName(agent)}-runs`, 'agent-runs');

export const isDefaultExperiment = (experiment: Pick<ExperimentResponse, 'metadata'>): boolean =>
  !!experiment.metadata?.[DEFAULT_EXPERIMENT_AGENT_KEY];

const isNotFound = (err: unknown): boolean =>
  (err as { response?: { status?: number } })?.response?.status === 404;

/** The agent's default experiment, created on first use. A concurrent create is resolved by
 *  reading back whichever request won. */
export const ensureDefaultExperiment = async (
  workspace: string,
  agent: string,
  signal?: AbortSignal
): Promise<ExperimentResponse> => {
  const name = defaultExperimentName(agent);
  try {
    return await getExperiment(workspace, name, signal);
  } catch (err) {
    if (!isNotFound(err)) throw err;
  }
  try {
    return await createExperiment(
      workspace,
      {
        name,
        description: `Runs of ${bareName(agent)} not yet linked to an experiment.`,
        metadata: { [DEFAULT_EXPERIMENT_AGENT_KEY]: bareName(agent) },
      },
      signal
    );
  } catch (err) {
    if (isConflictError(err)) return getExperiment(workspace, name, signal);
    throw err;
  }
};
