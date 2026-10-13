// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { jobsGetJob } from '@nemo/sdk/generated/platform/jobs';
import { Banner, Button, Card, Flex, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import type { AssistantToolArgs } from '@studio/routes/agents/AssistantChatRoute/toolParts';
import {
  getJobProgressDetailRoute,
  getJobProgressRefetchInterval,
  getStringArg,
  isJobProgressNotFoundError,
} from '@studio/routes/agents/AssistantChatRoute/utils/jobProgress';
import { useQuery } from '@tanstack/react-query';
import { ClipboardList } from 'lucide-react';
import { type FC } from 'react';
import { Link } from 'react-router';

interface JobProgressToolCallProps {
  readonly args: AssistantToolArgs;
}

export const JobProgressToolCall: FC<JobProgressToolCallProps> = ({ args }) => {
  const routeWorkspace = useWorkspaceFromPath();
  const workspace = getStringArg(args, 'workspace') ?? routeWorkspace;
  const jobName = getStringArg(args, 'job_name') ?? getStringArg(args, 'name');
  const jobType = getStringArg(args, 'job_type') ?? getStringArg(args, 'type');
  const source = getStringArg(args, 'source');
  const fallbackTitle = getStringArg(args, 'title');
  const fallbackDescription = getStringArg(args, 'description');

  const {
    data: job,
    error,
    isLoading,
  } = useQuery({
    queryKey: ['assistant', 'job-progress', workspace, jobName],
    queryFn: ({ signal }) => jobsGetJob(workspace, jobName ?? '', signal),
    enabled: !!workspace && !!jobName,
    refetchInterval: (query) =>
      getJobProgressRefetchInterval({
        status: query.state.data?.status,
        jobMissing: isJobProgressNotFoundError(query.state.error),
      }),
    retry: (failureCount, queryError) =>
      !isJobProgressNotFoundError(queryError) && failureCount < 3,
  });

  const isMissingJob = isJobProgressNotFoundError(error);
  const displayName = job?.name ?? jobName ?? fallbackTitle ?? 'Job progress';
  const description = job?.description || fallbackDescription || job?.source || source;
  const detailRoute = jobName
    ? getJobProgressDetailRoute({ job, jobName, jobType, source, workspace })
    : undefined;

  return (
    <Card
      className="my-density-md h-fit w-full max-w-full border border-base bg-surface-raised shadow-none"
      data-testid="job-progress-tool-call"
    >
      <Stack gap="density-md">
        <Flex align="center" justify="between" gap="density-md" wrap="wrap">
          <Flex align="center" gap="density-sm" className="min-w-0">
            <Flex
              align="center"
              justify="center"
              className="size-8 shrink-0 rounded bg-surface-sunken text-secondary"
            >
              <ClipboardList aria-hidden className="size-4" />
            </Flex>
            <Stack gap="density-xs" className="min-w-0">
              <Text kind="label/bold/md" className="truncate">
                {displayName}
              </Text>
              <Text kind="body/regular/sm" className="truncate text-secondary">
                {description ?? 'Job progress'}
              </Text>
            </Stack>
          </Flex>
          {job ? <StatusBadge status={job.status} /> : null}
        </Flex>

        {isLoading ? (
          <Flex align="center" gap="density-sm">
            <Spinner size="small" aria-label="Loading job..." />
            <Text className="text-secondary" kind="body/regular/sm">
              Loading job...
            </Text>
          </Flex>
        ) : null}

        {error && !isMissingJob ? (
          <Banner kind="inline" status="error">
            {error instanceof Error ? error.message : 'Failed to load job'}
          </Banner>
        ) : null}

        {!isLoading && jobName && isMissingJob ? (
          <Banner kind="inline" status="warning">
            Job "{jobName}" could not be found.
          </Banner>
        ) : null}

        {detailRoute ? (
          <Flex justify="end">
            <Button kind="secondary" asChild>
              <Link to={detailRoute}>View details</Link>
            </Button>
          </Flex>
        ) : null}
      </Stack>
    </Card>
  );
};
