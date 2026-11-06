// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_LOGS_PREFIX } from '@studio/components/evaluation/AgentEvaluationWizard/consts';
import { parquetBatchGroups } from '@studio/components/evaluation/AgentEvaluationWizard/parquetBatchGroups';
import type { WizardDataset } from '@studio/components/evaluation/AgentEvaluationWizard/types';
import { BUILDER_CONFIG_FILENAME } from '@studio/routes/DataDesignerJobDetailsRoute/builderConfig';

const EDITED_DATASET = /\.edited\.jsonl$/i;
const DATASET_FILE = /\.(parquet|jsonl|json|csv)$/i;

/**
 * The dataset a job's output fileset holds, as the evaluation wizard reads it: the user's edits
 * when the Data tab saved some, else every Parquet batch together, else the one data file.
 */
export const evaluationDatasetFromOutput = (
  fileset: string,
  paths: readonly string[]
): WizardDataset | undefined => {
  const dataPaths = paths.filter(
    (path) =>
      !path.startsWith(JOB_LOGS_PREFIX) &&
      !path.endsWith(BUILDER_CONFIG_FILENAME) &&
      DATASET_FILE.test(path)
  );
  const edited = dataPaths.find((path) => EDITED_DATASET.test(path));
  if (edited) return { fileset, file: edited };
  const [batches] = parquetBatchGroups(dataPaths);
  if (batches) return { fileset, file: batches.firstPath, batchGlob: batches.glob };
  const file = dataPaths.find((path) => path.endsWith('.parquet')) ?? dataPaths[0];
  return file ? { fileset, file } : undefined;
};
