// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createEvaluation } from '@nemo/sdk/generated/platform/evaluations';
import { filesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import type { EvaluationResponse } from '@nemo/sdk/generated/platform/schema';
import {
  buildEvalJobName,
  type EvalConfigFormat,
} from '@studio/components/evaluation/submitEvaluationJob';

/** Evaluation metadata key holding the name of the fileset that stores its eval config.
 *  Lives on the Evaluation, not its ExperimentGroup: the config is a property of a single
 *  run, and the Evaluation entity's ``metadata`` is documented for exactly this "config
 *  snapshot". Metadata values are plain strings, which is all a fileset name needs to be. */
export const EVAL_CONFIG_FILESET_KEY = 'eval_config_fileset';

/** Flat filename the reusable config is stored as inside its fileset. Every reusable
 *  evaluation's fileset carries one at the root — there is no per-run file picker. The
 *  extension records the serialization its author uploaded. */
export const evalConfigFilename = (format: EvalConfigFormat): string => `eval-config.${format}`;

/** Names a stored config may go by, newest convention first. ``.yml`` is read but never
 *  written, so a fileset seeded by hand or by an older client still resolves. */
export const EVAL_CONFIG_FILENAMES = [
  evalConfigFilename('json'),
  evalConfigFilename('yaml'),
  'eval-config.yml',
] as const;

/** The config file a fileset actually carries at its root, or null when it carries none. */
export const findEvalConfigFile = async (
  workspace: string,
  filesetName: string,
  signal?: AbortSignal
): Promise<string | null | undefined> => {
  const files = await filesListFilesetFiles(workspace, filesetName, undefined, signal).catch(
    () => null
  );
  if (!files) return undefined;
  const paths = new Set((files.data ?? []).map((file) => file.path));
  return EVAL_CONFIG_FILENAMES.find((name) => paths.has(name)) ?? null;
};

/** The fileset an Evaluation stores its eval config in, or null when it names none. A blank
 *  or whitespace-only metadata value counts as "none": it would otherwise pass the picker's
 *  filter and even become the default, only to be rejected by evaluationConfigError.
 *  This is stored by Studio UI convention only, not enforced by API or CLI */
export const evaluationFilesetName = (evaluation: EvaluationResponse): string | null => {
  const name = evaluation.metadata?.[EVAL_CONFIG_FILESET_KEY]?.trim();
  return name ? name : null;
};

/** Why an Evaluation cannot be reused, or null when it can. Two ways to be invalid:
 *  it names no fileset, or the fileset it names has no eval config at the root. */
export const evaluationConfigError = async (
  workspace: string,
  evaluation: EvaluationResponse,
  signal?: AbortSignal
): Promise<string | null> => {
  const filesetName = evaluationFilesetName(evaluation);
  if (!filesetName) {
    return `Evaluation "${evaluation.name}" has no eval config fileset. Pick another evaluation, or create one from a template.`;
  }
  const configFile = await findEvalConfigFile(workspace, filesetName, signal);
  if (configFile === undefined) {
    return `Could not read fileset "${filesetName}" for evaluation "${evaluation.name}". Pick another evaluation.`;
  }
  return configFile
    ? null
    : `Fileset "${filesetName}" has no ${EVAL_CONFIG_FILENAMES.join(' or ')} at its root, so evaluation "${evaluation.name}" cannot be reused. Pick another evaluation.`;
};

/** Create the Intake Evaluation this run publishes under, returning its **name** —
 *  which is what ``publication.intake.evaluation_id`` takes (the entity id is not it).
 *  The eval-config fileset pointer is written to the Evaluation's own ``metadata`` (the
 *  documented "config snapshot" home) so a later run can reuse it; ``parentEvaluationId``
 *  records the evaluation this one was derived from, when reusing an existing one.
 *  ``name`` is the user-supplied record name where there is one; without it the name is
 *  generated from ``nameStem`` with a random suffix so repeat runs do not collide. */
export const createRunEvaluation = async (
  workspace: string,
  {
    experimentIds,
    name: explicitName,
    nameStem,
    filesetName,
    parentEvaluationId,
    signal,
  }: {
    experimentIds: string[];
    name?: string;
    nameStem: string;
    filesetName: string;
    parentEvaluationId?: string;
    signal?: AbortSignal;
  }
): Promise<string> => {
  const name = explicitName || buildEvalJobName(nameStem);
  await createEvaluation(
    workspace,
    {
      name,
      experiment_ids: experimentIds,
      dataset_name: filesetName,
      metadata: { [EVAL_CONFIG_FILESET_KEY]: filesetName },
      ...(parentEvaluationId ? { parent_evaluation_id: parentEvaluationId } : {}),
    },
    signal
  );
  return name;
};
