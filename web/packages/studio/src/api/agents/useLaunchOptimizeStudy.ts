// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isNotFoundError } from '@nemo/common/src/api/common/utils';
import { FILESET_NAME_MAX_LENGTH } from '@nemo/common/src/utils/filesetName';
import { agentOptimizationCreateRunStrategyJob } from '@nemo/sdk/generated/agent-optimization/agent-optimization';
import type { RunStrategyJob } from '@nemo/sdk/generated/agent-optimization/schema/RunStrategyJob';
import {
  filesCreateFileset,
  filesDeleteFileset,
  filesRetrieveFileset,
} from '@nemo/sdk/generated/platform/files';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { rollbackFileset } from '@studio/api/agents/agentSpecFileset';
import type { BundleSource } from '@studio/api/files/types';
import { uploadFilesetEntries } from '@studio/api/files/uploadFilesetEntries';
import { type UseMutationOptions, useMutation } from '@tanstack/react-query';

/**
 * A bundle of `entries` is staged into a new fileset the study owns, so deleting the study deletes
 * it too. A `fileset` bundle is run from in place and never deleted with the study.
 */
export type LaunchOptimizeStudyParams = BundleSource & {
  workspace: string;
  agentName: string;
  /** The optimize YAML, as a path relative to the bundle root. */
  optimizeConfig: string;
  /** Name for the study; the server generates one when omitted. */
  name?: string;
};

/**
 * The optimization strategy Studio submits a staged bundle to.
 *
 * `run-strategy` is strategy-agnostic, but the bundle Studio stages here — an optimize YAML plus
 * the assets it references — is the input the `legacy` strategy takes, so the choice is fixed
 * rather than offered as a picker.
 */
export const STUDIO_OPTIMIZE_STRATEGY = 'legacy';

// Marks a bundle fileset that Studio created for exactly one study, so deleting the study may delete it.
const STUDIO_BUNDLE_FIELD = 'studio_bundle_fileset';

/**
 * Stamped on the bundle fileset itself at creation, carrying the agent it was staged for.
 *
 * The job-side {@link STUDIO_BUNDLE_FIELD} is only a pointer, and anyone who can submit a study can
 * write it — on its own it can name any fileset in the workspace, including a curated dataset
 * Studio never staged. This stamp lives on the fileset that is about to be deleted, so writing it
 * takes the same access that deleting the fileset takes: it cannot be used to aim a study delete at
 * data the study does not own.
 */
const STUDIO_BUNDLE_STAMP = 'studio_optimize_bundle';

export const studioBundleFileset = (job: RunStrategyJob): string | undefined => {
  const name = job.custom_fields?.[STUDIO_BUNDLE_FIELD];
  return typeof name === 'string' && name ? name : undefined;
};

/** Does the study actually run from this fileset, or does it merely claim it? */
const runsFromFileset = (job: RunStrategyJob, workspace: string, filesetName: string): boolean => {
  const configFileset = job.spec?.optimize_config_fileset;
  return configFileset === filesetName || configFileset === `${workspace}/${filesetName}`;
};

/** Did Studio stage this fileset as one study's disposable bundle? */
const isStagedBundle = (fileset: FilesetOutput): boolean =>
  typeof fileset.custom_fields?.[STUDIO_BUNDLE_STAMP] === 'string';

/**
 * Delete the bundle fileset Studio staged for a study, when the study owns one.
 *
 * Call this *before* deleting the study itself: the study's `custom_fields` is the only record of
 * the bundle's name, so a cleanup that fails once the study is gone leaves an orphaned fileset with
 * nothing left to retry from. Failing here instead keeps the pair intact and retryable, at the cost
 * of leaving the study in place. A fileset that is already gone counts as cleaned up, so a delete
 * that got half-way through can be retried.
 *
 * Both halves of the claim are checked before anything is deleted — the study must run from the
 * fileset it points at, and the fileset must carry Studio's {@link STUDIO_BUNDLE_STAMP} — because a
 * study is free-form client data. An unverifiable claim leaves the fileset in place: leaking a
 * bundle is recoverable, deleting a fileset this study never owned is not.
 */
export const deleteStudioBundleFileset = async (
  workspace: string,
  job: RunStrategyJob
): Promise<void> => {
  const filesetName = studioBundleFileset(job);
  if (!filesetName || !runsFromFileset(job, workspace, filesetName)) return;

  try {
    if (!isStagedBundle(await filesRetrieveFileset(workspace, filesetName))) return;
    await filesDeleteFileset(workspace, filesetName);
  } catch (error) {
    if (isNotFoundError(error)) return;
    throw new Error(
      `Could not delete the bundle fileset "${filesetName}" for this study, so the study was kept. ` +
        'Try deleting it again, or remove the fileset from Files.',
      { cause: error }
    );
  }
};

// Every launch stages its own bundle, so re-running a study never mutates the inputs of an earlier one.
// The agent segment is truncated, not the suffix, so a long agent name still fits a fileset name.
export const optimizeBundleFilesetName = (agentName: string, now = Date.now()): string => {
  const suffix = `-optimize-${now.toString(36)}`;
  const base = agentName.slice(0, FILESET_NAME_MAX_LENGTH - suffix.length).replace(/-+$/, '');
  return `${base}${suffix}`;
};

const createStudy = (
  { workspace, agentName, optimizeConfig, name }: LaunchOptimizeStudyParams,
  filesetName: string,
  customFields?: Record<string, string>
): Promise<RunStrategyJob> =>
  agentOptimizationCreateRunStrategyJob(workspace, {
    ...(name ? { name } : {}),
    spec: {
      strategy: STUDIO_OPTIMIZE_STRATEGY,
      optimize_config: optimizeConfig,
      optimize_config_fileset: `${workspace}/${filesetName}`,
      agent: agentName,
    },
    ...(customFields ? { custom_fields: customFields } : {}),
  });

export const launchOptimizeStudy = async (
  params: LaunchOptimizeStudyParams
): Promise<RunStrategyJob> => {
  const { workspace, agentName } = params;
  // No ownership marker: the fileset is the user's, so deleting the study must leave it alone.
  if (params.fileset !== undefined) return createStudy(params, params.fileset);

  const filesetName = optimizeBundleFilesetName(agentName);

  await filesCreateFileset(workspace, {
    name: filesetName,
    description: `Optimize bundle for ${agentName}`,
    custom_fields: { [STUDIO_BUNDLE_STAMP]: agentName },
  });

  try {
    await uploadFilesetEntries(workspace, filesetName, params.entries);
    return await createStudy(params, filesetName, { [STUDIO_BUNDLE_FIELD]: filesetName });
  } catch (error) {
    await rollbackFileset(workspace, filesetName);
    throw error;
  }
};

export type UseLaunchOptimizeStudyOptions = Omit<
  UseMutationOptions<RunStrategyJob, Error, LaunchOptimizeStudyParams>,
  'mutationFn'
>;

export const useLaunchOptimizeStudy = (options?: UseLaunchOptimizeStudyOptions) =>
  useMutation({ ...options, mutationFn: launchOptimizeStudy });
