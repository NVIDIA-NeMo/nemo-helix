// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { jsonFilter } from '@nemo/common/src/api/filterOperators';
import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import {
  agentsCreatePackageJob,
  agentsDownloadPackageJobResult,
  agentsGetPackageJobStatus,
  agentsListPackageJobs,
} from '@nemo/sdk/generated/agents/agents';
import {
  PackageAgentJobsSortField,
  type PackageAgentInput,
  type PackageAgentJobsListFilter,
} from '@nemo/sdk/generated/agents/schema';
import {
  isQueuedTooLong,
  isTerminalPackageStatus,
  PACKAGE_RESULT_NAME,
  parseJobTimestamp,
  parsePackageResult,
} from '@studio/api/agents/packageAgent';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useState } from 'react';

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
  const [submittedAt, setSubmittedAt] = useState<number | undefined>();

  // A reload loses the submitted job name, but not the job: without this the
  // page forgets an image the user watched being built.
  const lastJob = useQuery({
    queryKey: ['agents', 'package-job', workspace, agentName, 'latest'],
    queryFn: async () => {
      const page = await agentsListPackageJobs(workspace, {
        page_size: 1,
        sort: PackageAgentJobsSortField['-created_at'],
        filter: jsonFilter<PackageAgentJobsListFilter>({ 'spec.agent': { $eq: agentName } }),
      });
      const job = page.data[0];
      if (!job?.name) return null;
      return { name: job.name, createdAt: parseJobTimestamp(job.created_at) };
    },
    enabled: !jobName,
  });

  const activeJobName = jobName ?? lastJob.data?.name;

  const submit = useMutation({
    mutationFn: (input: Omit<PackageAgentInput, 'agent'> = {}) =>
      agentsCreatePackageJob(workspace, { spec: { agent: agentName, ...input } }),
    onSuccess: (job) => {
      setJobName(job.name);
      setSubmittedAt(Date.now());
    },
  });

  const status = useQuery({
    queryKey: ['agents', 'package-job', workspace, activeJobName, 'status'],
    queryFn: ({ signal }) => agentsGetPackageJobStatus(workspace, activeJobName ?? '', signal),
    enabled: Boolean(activeJobName),
    refetchInterval: (query) =>
      isTerminalPackageStatus(query.state.data?.status) ? false : JOB_POLLING_INTERVAL_MS,
  });

  const jobStatus = status.data?.status;
  // Without this the job is "still running" forever and the user cannot retry.
  const isUnreachable = status.isError;
  const isComplete = jobStatus === 'completed';

  const result = useQuery({
    queryKey: ['agents', 'package-job', workspace, activeJobName, 'result'],
    queryFn: async ({ signal }) => {
      const blob = await agentsDownloadPackageJobResult(
        workspace,
        activeJobName ?? '',
        PACKAGE_RESULT_NAME,
        signal
      );
      const parsed: unknown = JSON.parse(await blob.text());
      // react-query rejects an undefined queryFn result, and a job can finish
      // without a usable tag.
      return parsePackageResult(parsed) ?? null;
    },
    enabled: Boolean(activeJobName) && isComplete,
  });

  // The status poll doubles as the clock, so no extra timer is needed. A
  // restored job has no submit time, so it falls back to when it was created —
  // otherwise the stall warning never survives the reload it exists for.
  const startedAt = submittedAt ?? lastJob.data?.createdAt;

  return {
    /** Start a build. Extra `PackageAgentInput` fields are optional overrides. */
    packageAgent: submit.mutate,
    /** Rejected at submit time — a non-Fabric agent, or no host build environment. */
    submitError: submit.error,
    isSubmitting: submit.isPending,
    jobName: activeJobName,
    isRunning: Boolean(activeJobName) && !isUnreachable && !isTerminalPackageStatus(jobStatus),
    /** The status poll is failing, so the build can no longer be followed. */
    isUnreachable,
    /** The build predates this page load, so its progress was not watched here. */
    isRestored: !jobName && Boolean(lastJob.data),
    /** When a restored build ran, so a months-old tag does not read as fresh. */
    restoredAt: !jobName ? lastJob.data?.createdAt : undefined,
    isQueued: jobStatus === 'created' && !isUnreachable,
    /** Accepted but never dispatched — nothing is running the job. */
    isStalled: isQueuedTooLong(jobStatus, startedAt, status.dataUpdatedAt),
    isComplete,
    isFailed: jobStatus === 'error' || jobStatus === 'cancelled',
    /** The job is done but its result artifact has not been read yet. */
    isResultPending: isComplete && !result.isFetched,
    /** The artifact could not be read — distinct from a job that reported no tag. */
    resultError: result.isError,
    image: result.data?.image,
    published: result.data?.published,
  };
};
