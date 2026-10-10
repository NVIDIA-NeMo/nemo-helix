// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import type { EvaluationResponse, ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { EVAL_CONFIG_FILESET_KEY } from '@studio/components/evaluation/experimentEvalConfig';

export const experimentFixture = (id: string, name: string): ExperimentResponse => ({
  id,
  name,
  workspace: DEFAULT_WORKSPACE,
  default_sort: '-created_at',
  evaluation_count: 1,
});

/** An evaluation Studio can re-run: it points at its eval-config fileset. */
export const reusableEvaluationFixture = (
  name: string,
  experimentId: string,
  agent: string
): EvaluationResponse => ({
  id: `eval_${name}`,
  name,
  workspace: DEFAULT_WORKSPACE,
  experiment_ids: [experimentId],
  experiment_group_id: experimentId,
  dataset_name: 'ds',
  agent_names: [agent],
  metadata: { [EVAL_CONFIG_FILESET_KEY]: `${name}-data` },
});
