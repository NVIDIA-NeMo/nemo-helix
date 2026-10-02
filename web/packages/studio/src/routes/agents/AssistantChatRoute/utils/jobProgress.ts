// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getJobRefetchInterval } from '@nemo/common/src/utils/query';
import type { HelixJobResponse, HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { getJobDetailRoute } from '@studio/components/dataViews/JobsDataView/utils';
import {
  JOB_PROGRESS_JOB_TYPE,
  JOB_PROGRESS_JOB_TYPE_SOURCE,
} from '@studio/routes/agents/AssistantChatRoute/jobProgressConsts';
import { getAgentEvaluationDetailRoute } from '@studio/routes/utils';

interface JobProgressRefetchIntervalArgs {
  readonly jobMissing: boolean;
  readonly status?: HelixJobStatus;
}

interface JobProgressDetailRouteArgs {
  readonly job?: HelixJobResponse | null;
  readonly jobName: string;
  readonly jobType?: string;
  readonly source?: string;
  readonly workspace: string;
}

export const isJobProgressNotFoundError = (error: unknown): boolean => {
  if (typeof error !== 'object' || error === null) return false;
  const maybeError = error as {
    readonly response?: { readonly status?: number };
    readonly status?: number;
  };
  return maybeError.status === 404 || maybeError.response?.status === 404;
};

const normalizeLookupValue = (value: string | undefined): string | undefined => {
  const normalized = value?.trim().toLowerCase();
  return normalized || undefined;
};

export const getStringArg = (args: Record<string, unknown>, key: string): string | undefined => {
  const value = args[key];
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  return trimmed || undefined;
};

export const getJobProgressRefetchInterval = ({
  jobMissing,
  status,
}: JobProgressRefetchIntervalArgs): number | false => {
  if (jobMissing) return false;
  return getJobRefetchInterval(status);
};

export const getJobProgressDetailRoute = ({
  job,
  jobName,
  jobType,
  source,
  workspace,
}: JobProgressDetailRouteArgs): string => {
  const normalizedJobType = normalizeLookupValue(jobType);
  if (normalizedJobType === JOB_PROGRESS_JOB_TYPE.AGENT_EVALUATION) {
    return getAgentEvaluationDetailRoute(workspace, jobName);
  }

  const sourceOverride =
    normalizeLookupValue(source) ??
    (normalizedJobType ? JOB_PROGRESS_JOB_TYPE_SOURCE[normalizedJobType] : undefined);

  return getJobDetailRoute(
    {
      name: job?.name ?? jobName,
      source: job?.source ?? sourceOverride,
    },
    workspace
  );
};
