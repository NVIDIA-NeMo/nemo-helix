// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_SOURCE } from '@studio/components/dataViews/JobsDataView/constants';
import {
  CUSTOMIZER_ENABLED,
  DATA_DESIGNER_ENABLED,
  EVALS_ENABLED,
  SAFE_SYNTHESIZER_ENABLED,
} from '@studio/constants/environment';
import {
  getDataDesignerJobDetailsRoute,
  getEvaluationResultDetailsRoute,
  getGenerateJobRoute,
  getWorkspaceCustomizationJobDetailsRoute,
  getWorkspaceJobDetailRoute,
} from '@studio/routes/utils';

interface JobDetailRouteJob {
  readonly name: string;
  readonly source?: string | null;
}

const SOURCE_DETAIL_ROUTE: Record<
  string,
  { readonly enabled: boolean; readonly getRoute: (workspace: string, jobName: string) => string }
> = {
  [JOB_SOURCE.CUSTOMIZATION]: {
    enabled: CUSTOMIZER_ENABLED,
    getRoute: getWorkspaceCustomizationJobDetailsRoute,
  },
  [JOB_SOURCE.DATA_DESIGNER]: {
    enabled: DATA_DESIGNER_ENABLED,
    getRoute: getDataDesignerJobDetailsRoute,
  },
  [JOB_SOURCE.SAFE_SYNTHESIZER]: {
    enabled: SAFE_SYNTHESIZER_ENABLED,
    getRoute: getGenerateJobRoute,
  },
  [JOB_SOURCE.EVALUATOR_METRICS]: {
    enabled: EVALS_ENABLED,
    getRoute: getEvaluationResultDetailsRoute,
  },
};

export const getJobDetailRoute = (job: JobDetailRouteJob, workspace: string): string => {
  const genericRoute = getWorkspaceJobDetailRoute(workspace, job.name);
  const entry = job.source ? SOURCE_DETAIL_ROUTE[job.source] : undefined;
  return entry?.enabled ? entry.getRoute(workspace, job.name) : genericRoute;
};
