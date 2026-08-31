// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import {
  agentsCreatePackageJob,
  agentsDownloadPackageJobResult,
  agentsGetPackageJobLogs,
  agentsGetPackageJobStatus,
} from '@nemo/sdk/generated/agents/api';
import type { PackageAgentInput } from '@nemo/sdk/generated/agents/schema';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useState } from 'react';

/** Only Platform-managed agents can be packaged; NAT workflows build from a checkout. */
export const FABRIC_CONFIG_FORMAT = 'nemo-agents-spec-v1';

/** Name the packaging job saves its result under; see `PACKAGE_RESULT_NAME` in the plugin. */
export const PACKAGE_RESULT_NAME = 'package_result';

/** Statuses that will not change again, so polling can stop. */
const TERMINAL_JOB_STATUSES = new Set(['completed', 'error', 'cancelled']);

export const isTerminalPackageStatus = (status: string | undefined): boolean =>
  status !== undefined && TERMINAL_JOB_STATUSES.has(status);

/** The tag to hand to a deployment, plus the remote reference when the job pushed one. */
export interface PackageResult {
  image: string;
  agent: string;
  published: string;
}

/**
 * Narrow the `package_result` artifact, which is untrusted JSON off the wire.
 *
 * A result without a usable `image` is treated as absent rather than surfaced
 * as a blank tag the user could paste into a deployment.
 */
export const parsePackageResult = (parsed: unknown): PackageResult | undefined => {
  const image = (parsed as { image?: unknown })?.image;
  if (typeof image !== 'string' || !image) return undefined;
  const agent = (parsed as { agent?: unknown })?.agent;
  const published = (parsed as { published?: unknown })?.published;
  return {
    image,
    agent: typeof agent === 'string' ? agent : '',
    published: typeof published === 'string' ? published : '',
  };
};

interface UsePackageAgentParams {
  workspace: string;
  agentName: string;
}

/**
 * Submit a packaging job for *agentName* and follow it to a terminal status.
 *
 * The image tag is not on the job row — the task writes it to the
 * `package_result` artifact, so it is fetched separately once the job completes.
 */
export const usePackageAgent = ({ workspace, agentName }: UsePackageAgentParams) => {
  const [jobName, setJobName] = useState<string | undefined>();

  const submit = useMutation({
    mutationFn: (input: Omit<PackageAgentInput, 'agent'> = {}) =>
      agentsCreatePackageJob(workspace, { spec: { agent: agentName, ...input } }),
    onSuccess: (job) => setJobName(job.name),
  });

  const status = useQuery({
    queryKey: ['agents', 'package-job', workspace, jobName, 'status'],
    queryFn: () => agentsGetPackageJobStatus(workspace, jobName as string),
    enabled: Boolean(jobName),
    refetchInterval: (query) =>
      isTerminalPackageStatus(query.state.data?.status) ? false : JOB_POLLING_INTERVAL_MS,
  });

  const jobStatus = status.data?.status;
  const isComplete = jobStatus === 'completed';

  // `jobStatus` is in the key so the flip to terminal fetches once more: the
  // lines explaining a failure land after the status changes, and polling has
  // stopped by then.
  const logs = useQuery({
    queryKey: ['agents', 'package-job', workspace, jobName, 'logs', jobStatus],
    queryFn: () => agentsGetPackageJobLogs(workspace, jobName as string),
    enabled: Boolean(jobName),
    refetchInterval: isTerminalPackageStatus(jobStatus) ? false : JOB_POLLING_INTERVAL_MS,
  });

  const result = useQuery({
    queryKey: ['agents', 'package-job', workspace, jobName, 'result'],
    queryFn: async () => {
      const blob = await agentsDownloadPackageJobResult(
        workspace,
        jobName as string,
        PACKAGE_RESULT_NAME
      );
      const parsed: unknown = JSON.parse(await blob.text());
      // react-query rejects an undefined queryFn result, and a job can finish
      // without a usable tag.
      return parsePackageResult(parsed) ?? null;
    },
    enabled: Boolean(jobName) && isComplete,
  });

  return {
    /** Start a build. Extra `PackageAgentInput` fields are optional overrides. */
    packageAgent: submit.mutate,
    /** Rejected at submit time — a non-Fabric agent, or no host build environment. */
    submitError: submit.error,
    isSubmitting: submit.isPending,
    jobName,
    jobStatus,
    isRunning: Boolean(jobName) && !isTerminalPackageStatus(jobStatus),
    isComplete,
    isFailed: jobStatus === 'error' || jobStatus === 'cancelled',
    logs: logs.data?.data ?? [],
    isLogsLoading: logs.isLoading,
    image: result.data?.image,
    published: result.data?.published,
    reset: () => setJobName(undefined),
  };
};
