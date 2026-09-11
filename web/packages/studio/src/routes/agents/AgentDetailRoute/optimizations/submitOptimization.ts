// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { agentsCreateOptimizeJob } from '@nemo/sdk/generated/agents/agents';
import type { OptimizeJob } from '@nemo/sdk/generated/agents/schema';
import { ensureEvalConfigFileset } from '@studio/api/evaluation/eval-config-fileset';
import {
  STUDY_DATASET_PATH,
  type StudyDatasetRow,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/studyDataset';

/** Flat path the generated config is stored under inside its fileset. `spec.optimize_config` is
 *  resolved relative to the fileset root, so a bare filename is all the job needs. */
export const OPTIMIZE_CONFIG_PATH = 'optimize.yaml';

/** One fileset per study, named after it. Job names are unique per workspace, so this never collides with another study's bundle. */
export const optimizeFilesetName = (jobName: string): string => `${jobName}-optimize`;

interface SubmitOptimizationInput {
  workspace: string;
  agentName: string;
  /** Already-sanitized job name; also names the fileset. */
  name: string;
  config: string;
  /** Rows the study replays, staged beside the config so `eval.general.dataset` resolves. */
  dataset: StudyDatasetRow[];
  signal?: AbortSignal;
}

/** Stage the generated config and submit the study. */
export const submitOptimization = async ({
  workspace,
  agentName,
  name,
  config,
  dataset,
  signal,
}: SubmitOptimizationInput): Promise<OptimizeJob> => {
  const fileset = optimizeFilesetName(name);

  await ensureEvalConfigFileset(
    workspace,
    fileset,
    signal ?? new AbortController().signal,
    [
      { path: OPTIMIZE_CONFIG_PATH, content: config, type: 'application/yaml' },
      {
        path: STUDY_DATASET_PATH,
        content: JSON.stringify(dataset, null, 2),
        type: 'application/json',
      },
    ],
    `Optimization bundle for ${agentName}`
  );

  return agentsCreateOptimizeJob(
    workspace,
    {
      name,
      spec: {
        optimize_config: OPTIMIZE_CONFIG_PATH,
        optimize_config_fileset: fileset,
        workspace,
        agent: agentName,
      },
    },
    signal
  );
};
