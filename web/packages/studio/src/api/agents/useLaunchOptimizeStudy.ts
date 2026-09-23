// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { agentsCreateOptimizeJob } from '@nemo/sdk/generated/agents/agents';
import type { OptimizeJob } from '@nemo/sdk/generated/agents/schema/OptimizeJob';
import { filesCreateFileset } from '@nemo/sdk/generated/platform/files';
import { rollbackFileset } from '@studio/api/agents/agentSpecFileset';
import { type FilesetEntry, uploadFilesetEntries } from '@studio/api/files/uploadFilesetEntries';
import { type UseMutationOptions, useMutation } from '@tanstack/react-query';

export interface LaunchOptimizeStudyParams {
  workspace: string;
  agentName: string;
  entries: readonly FilesetEntry[];
  /** The optimize YAML, as a path relative to the bundle root. */
  optimizeConfig: string;
}

// Every launch stages its own bundle, so re-running a study never mutates the inputs of an earlier one.
export const optimizeBundleFilesetName = (agentName: string, now = Date.now()): string =>
  `${agentName}-optimize-${now.toString(36)}`;

export const launchOptimizeStudy = async ({
  workspace,
  agentName,
  entries,
  optimizeConfig,
}: LaunchOptimizeStudyParams): Promise<OptimizeJob> => {
  const filesetName = optimizeBundleFilesetName(agentName);

  await filesCreateFileset(workspace, {
    name: filesetName,
    description: `Optimize bundle for ${agentName}`,
  });

  try {
    await uploadFilesetEntries(workspace, filesetName, entries);

    return await agentsCreateOptimizeJob(workspace, {
      spec: {
        optimize_config: optimizeConfig,
        optimize_config_fileset: `${workspace}/${filesetName}`,
        agent: agentName,
        workspace,
      },
    });
  } catch (error) {
    await rollbackFileset(workspace, filesetName);
    throw error;
  }
};

export type UseLaunchOptimizeStudyOptions = Omit<
  UseMutationOptions<OptimizeJob, Error, LaunchOptimizeStudyParams>,
  'mutationFn'
>;

export const useLaunchOptimizeStudy = (options?: UseLaunchOptimizeStudyOptions) =>
  useMutation({ ...options, mutationFn: launchOptimizeStudy });
