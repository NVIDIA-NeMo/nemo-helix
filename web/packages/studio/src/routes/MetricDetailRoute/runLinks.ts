// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { MetricRun } from '@studio/api/metrics/metricScores';
import {
  getAgentDetailRoute,
  getAgentEvaluationDetailRoute,
  getEvaluationResultDetailsRoute,
  getModelDetailRoute,
} from '@studio/routes/utils';

/** The run's own results page. Keyed by job, since a taskset run's name is not its job. */
export const runHref = (workspace: string, run: MetricRun): string =>
  run.kind === 'dataset'
    ? getEvaluationResultDetailsRoute(workspace, run.jobId)
    : getAgentEvaluationDetailRoute(workspace, run.jobId);

/** Agent targets include Fabric runners; a runner's own model has no page, so it gets no link. */
const AGENT_TARGET_KINDS = ['agent', 'fabric'];

export const targetHref = (workspace: string, run: MetricRun): string | undefined => {
  if (!run.targetName) return undefined;
  if (run.targetKind === 'model') {
    // A model ref may carry its workspace; the route wants the bare name.
    return getModelDetailRoute(workspace, run.targetName.split('/').pop() ?? run.targetName);
  }
  if (run.targetKind && AGENT_TARGET_KINDS.includes(run.targetKind)) {
    return getAgentDetailRoute(workspace, run.targetName);
  }
  return undefined;
};
