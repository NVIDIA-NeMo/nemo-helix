// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EvaluationResponse } from '@nemo/sdk/generated/platform/schema';
import {
  evaluationFilesetName,
  findEvalConfigFile,
} from '@studio/components/evaluation/experimentEvalConfig';

/** The exact config saved with the selected evaluation, as the `workspace/fileset#path` the study
 *  generates its dataset and scoring from. Intake omits the grading context, so it cannot. */
export const evaluationConfigRef = async (
  workspace: string,
  evaluation: EvaluationResponse
): Promise<string> => {
  const filesetName = evaluationFilesetName(evaluation);
  if (!filesetName) {
    throw new Error(
      `Evaluation "${evaluation.name}" has no stored eval config. Pick an evaluation run from Studio.`
    );
  }
  const configFile = await findEvalConfigFile(workspace, filesetName);
  if (!configFile)
    throw new Error(`Could not find the eval config for evaluation "${evaluation.name}".`);
  return `${workspace}/${filesetName}#${configFile}`;
};
