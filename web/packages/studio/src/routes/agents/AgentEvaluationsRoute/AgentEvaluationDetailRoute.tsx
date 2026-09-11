// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { AccordionPanel } from '@nemo/common/src/components/AccordionPanel';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { KVPair } from '@nemo/common/src/components/KVPair';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { useLiveSeconds } from '@nemo/common/src/hooks/useLiveSeconds';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { formatDurationMs, formatTimeInSeconds, utcToLocalDate } from '@nemo/common/src/utils/date';
import { evaluatorCancelAgentEvaluateJob } from '@nemo/sdk/generated/evaluator/evaluator-plugin-agent-eval-jobs-routes';
import { useGetEvaluation } from '@nemo/sdk/generated/platform/evaluations';
import type { PlatformJobStatus } from '@nemo/sdk/generated/platform/schema';
import {
  Block,
  Button,
  Flex,
  Badge,
  Grid,
  Modal,
  PageHeader,
  Panel,
  Spinner,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import {
  aggregateScoresOf,
  agentNameForJob,
  evalConfigName,
  fetchAgentEvalBundle,
  fetchAgentEvalJob,
  fetchAgentEvalResult,
  joinBundleByTask,
  parseBundleRef,
} from '@studio/api/evaluation/agent-evaluations';
import { evalDurationMs } from '@studio/api/evaluation/utils';
import { AgentEvalTaskResultsPanel } from '@studio/components/evaluation/AgentEvalTaskResultsPanel';
import { EvalAggregateScoresTable } from '@studio/components/evaluation/EvalAggregateScoresTable';
import { StatusLogsContent } from '@studio/components/evaluation/Jobs/StatusLogsContent';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import {
  getAgentEvaluationsTabRoute,
  getAgentsListRoute,
  getFilesetRoute,
} from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { CircleX, ClipboardList, FlaskConical, ScrollText } from 'lucide-react';
import { type FC, useEffect, useState } from 'react';
import { Link } from 'react-router';

const TERMINAL_STATUSES = new Set([
  'completed',
  'succeeded',
  'success',
  'failed',
  'cancelled',
  'canceled',
  'error',
]);

const isTerminal = (status: string | undefined): boolean =>
  TERMINAL_STATUSES.has((status ?? '').toLowerCase());

export const AgentEvaluationDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { agentEvalJobName: jobName } = useRequiredPathParams([ROUTE_PARAMS.agentEvalJobName]);
  const toast = useToast();
  const queryClient = useQueryClient();

  // Job + status — refetched while the job is non-terminal so the badge stays
  // live without forcing a page reload.
  const { data: job, isLoading: isLoadingJob } = useQuery({
    queryKey: ['agent-eval-job', workspace, jobName] as const,
    queryFn: ({ signal }) => fetchAgentEvalJob(workspace, jobName, signal),
    enabled: !!workspace && !!jobName,
    refetchInterval: (query) => (isTerminal(query.state.data?.status) ? false : 5_000),
  });

  // The "Evaluations" crumb links the agent's eval tab, but the agent name only arrives with
  // the loaded job — so set breadcrumbs from an effect keyed on it (the useBreadcrumbs `items`
  // param runs once on mount and would keep the crumb non-clickable after the job resolves).
  const agentName = job ? agentNameForJob(job) : null;
  const { setBreadcrumbs } = useBreadcrumbs();
  useEffect(() => {
    setBreadcrumbs([
      { slotLabel: 'Agents', href: getAgentsListRoute(workspace) },
      agentName
        ? {
            slotLabel: 'Evaluations',
            href: getAgentEvaluationsTabRoute(workspace, agentName),
            preserveQuery: true,
          }
        : { slotLabel: 'Evaluations' },
      { slotLabel: jobName },
    ]);
    return () => setBreadcrumbs([]);
  }, [setBreadcrumbs, workspace, agentName, jobName]);

  const isJobTerminal = isTerminal(job?.status);
  const canCancelJob = !!job?.status && !isJobTerminal;
  const [cancelModalOpen, setCancelModalOpen] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const liveSeconds = useLiveSeconds({
    startDate: !isJobTerminal ? utcToLocalDate(job?.created_at) : undefined,
  });

  // How long the run took, once it finished. The job row itself cannot answer this — it is written
  // at create and on rerun, never on a status change — so the duration comes from the evaluation the
  // run published under. Fetched by name, so it lands as soon as the publish does.
  const publishedEvaluation = job?.spec.publication?.intake?.evaluation_id;
  const { data: evaluation } = useGetEvaluation(workspace, publishedEvaluation ?? '', {
    query: { enabled: !!workspace && !!publishedEvaluation && isJobTerminal },
  });
  const durationMs = evalDurationMs(evaluation?.metadata);

  const handleCancelJob = async () => {
    if (!jobName) return;
    setIsCancelling(true);
    try {
      await evaluatorCancelAgentEvaluateJob(workspace, jobName);
      toast.success('Job cancelled');
      setCancelModalOpen(false);
      void queryClient.invalidateQueries({ queryKey: ['agent-eval-job', workspace, jobName] });
    } catch {
      toast.error('Failed to cancel job. Please try again.');
    } finally {
      setIsCancelling(false);
    }
  };

  // Aggregate scores (mean/min/max per metric) from the queryable result record.
  // Only meaningful once the job is terminal. The record is persisted best-effort and
  // may lag the terminal status, so poll while it is still absent and stop once it loads.
  const { data: result, isLoading: isLoadingResult } = useQuery({
    queryKey: ['agent-eval-result', workspace, jobName] as const,
    queryFn: ({ signal }) => fetchAgentEvalResult(workspace, jobName, signal),
    enabled: !!workspace && !!jobName && isJobTerminal,
    refetchInterval: (query) => (query.state.data == null ? 5_000 : false),
  });

  // Per-task detail (agent response + per-task score + diagnostics) from the
  // result bundle referenced by the record. Gated on the result being loaded; polls
  // while the bundle is still absent so late-written artifacts are picked up.
  const { data: bundle, isLoading: isLoadingBundle } = useQuery({
    queryKey: ['agent-eval-bundle', workspace, jobName, result?.bundle_ref] as const,
    queryFn: ({ signal }) => fetchAgentEvalBundle(workspace, result?.bundle_ref, signal),
    enabled: !!workspace && isJobTerminal && !!result?.bundle_ref,
    refetchInterval: (query) => (query.state.data == null ? 5_000 : false),
  });

  if (isLoadingJob && !job) {
    return (
      <Flex align="center" justify="center" className="h-full w-full">
        <Spinner size="medium" aria-label="Loading evaluation..." />
      </Flex>
    );
  }

  if (!job) {
    return (
      <Stack padding="density-2xl">
        <ErrorMessage
          header="Evaluation not found"
          message={`No evaluation job named "${jobName}" in workspace "${workspace}".`}
        />
      </Stack>
    );
  }

  const statusMessage =
    typeof job.status_details?.message === 'string' ? job.status_details.message : null;
  const errorMessage =
    typeof job.error_details?.message === 'string' ? job.error_details.message : null;
  const allScores = aggregateScoresOf(result ?? null);
  const nativeScores = allScores.filter((s) => !s.name.startsWith('runner.'));
  const runnerScores = allScores.filter((s) => s.name.startsWith('runner.'));
  const taskDetails = joinBundleByTask(bundle ?? null);
  const artifactsFileset = result?.bundle_ref
    ? (parseBundleRef(result.bundle_ref)?.fileset ?? null)
    : null;

  return (
    <AccessibleTitle title={`Evaluation - ${jobName}`}>
      <Stack className="w-full p-density-2xl min-h-full" gap="density-2xl">
        <PageHeader
          slotHeading={
            <Flex align="center" gap="2">
              {jobName}
              <Badge kind="outline" color="gray">
                Task-Driven
              </Badge>
            </Flex>
          }
          slotDescription="Agent evaluation via nemo-evaluator. Scores aggregate per metric across the evaluated tasks."
        />

        <Grid cols={{ base: 1, xl: 2 }} gap="density-2xl">
          <Panel
            elevation="high"
            density="compact"
            slotHeading={
              <Flex align="center" justify="between" className="w-full">
                <Flex align="center" gap="2">
                  <ClipboardList />
                  Details
                </Flex>
                {canCancelJob && (
                  <Button
                    kind="tertiary"
                    color="danger"
                    size="small"
                    onClick={() => setCancelModalOpen(true)}
                  >
                    <CircleX /> Cancel Job
                  </Button>
                )}
              </Flex>
            }
          >
            <Stack gap="density-xl">
              <KVPair label="Name" value={job.name} loading={isLoadingJob} />
              <KVPair
                label="Status"
                value={
                  <Flex align="center" gap="density-md">
                    <StatusBadge status={job.status} />
                    {!isJobTerminal && liveSeconds !== undefined && (
                      <Text kind="body/regular/sm">{formatTimeInSeconds(liveSeconds)}</Text>
                    )}
                    {isJobTerminal && durationMs !== undefined && (
                      <Text kind="body/regular/sm">{formatDurationMs(durationMs)}</Text>
                    )}
                  </Flex>
                }
                loading={isLoadingJob}
              />
              <KVPair label="Agent" value={agentNameForJob(job) ?? '-'} loading={isLoadingJob} />
              <KVPair
                label="Tasks"
                value={String(job.spec.tasks?.length ?? '-')}
                loading={isLoadingJob}
              />
              {evalConfigName(job) && (
                <KVPair
                  label="Eval Config"
                  value={
                    <Link
                      to={getFilesetRoute(workspace, evalConfigName(job) ?? '')}
                      className="text-primary underline"
                    >
                      {evalConfigName(job)}
                    </Link>
                  }
                />
              )}
              <KVPair
                label="Created"
                value={job.created_at ? <RelativeTime datetime={job.created_at} /> : ''}
                loading={isLoadingJob}
              />
              {artifactsFileset && (
                <KVPair
                  label="Artifacts"
                  value={
                    <Link
                      to={getFilesetRoute(workspace, artifactsFileset)}
                      className="text-primary underline"
                    >
                      View files
                    </Link>
                  }
                />
              )}
              {(errorMessage ?? statusMessage) && (
                <KVPair
                  label="Response"
                  value={
                    <Text kind="body/regular/sm" color={errorMessage ? 'danger' : 'default'}>
                      {errorMessage ?? statusMessage}
                    </Text>
                  }
                />
              )}
            </Stack>
          </Panel>

          <Panel
            slotHeading="Scores"
            slotIcon={<FlaskConical />}
            elevation="high"
            density="compact"
          >
            {!isJobTerminal && (
              <Block className="text-subtle">
                Scores are computed once the job reaches a terminal state.
              </Block>
            )}
            {isJobTerminal && isLoadingResult && (
              <Flex justify="center" align="center" className="min-h-[120px] w-full">
                <Spinner size="small" aria-label="Loading scores..." />
              </Flex>
            )}
            {isJobTerminal && !isLoadingResult && (
              <Stack gap="density-lg">
                <EvalAggregateScoresTable
                  scores={[
                    ...nativeScores.filter((s) => s.name.startsWith('view.')),
                    ...nativeScores.filter((s) => !s.name.startsWith('view.')),
                  ]}
                  emptyMessage={
                    runnerScores.length > 0
                      ? 'No native scores recorded for this evaluation.'
                      : undefined
                  }
                />
                {runnerScores.length > 0 && (
                  <Stack gap="density-sm">
                    <Text kind="body/semibold/md">Runner Scores</Text>
                    <EvalAggregateScoresTable scores={runnerScores} disableScoreColoring />
                  </Stack>
                )}
              </Stack>
            )}
          </Panel>
        </Grid>

        {isJobTerminal && isLoadingBundle && (
          <Flex justify="center" align="center" className="min-h-[120px] w-full">
            <Spinner size="small" aria-label="Loading task results..." />
          </Flex>
        )}

        {isJobTerminal && !isLoadingBundle && taskDetails.length > 0 && (
          <AgentEvalTaskResultsPanel tasks={taskDetails} />
        )}

        <AccordionPanel slotHeading="Logs" slotIcon={<ScrollText />}>
          <StatusLogsContent
            workspace={workspace}
            jobName={jobName}
            jobStatus={job.status as PlatformJobStatus}
          />
        </AccordionPanel>
      </Stack>

      <Modal
        open={cancelModalOpen}
        onOpenChange={(open) => !isCancelling && setCancelModalOpen(open)}
        slotHeading={
          <Flex align="center" gap="density-sm">
            <CircleX />
            Cancel Evaluation Job
          </Flex>
        }
        slotFooter={
          <Flex justify="end" gap="density-xs" align="center" className="w-full">
            <Button
              onClick={() => setCancelModalOpen(false)}
              kind="tertiary"
              color="neutral"
              disabled={isCancelling}
            >
              Go Back
            </Button>
            <Button color="danger" onClick={handleCancelJob} disabled={isCancelling}>
              {isCancelling ? 'Cancelling...' : 'Cancel Job'}
            </Button>
          </Flex>
        }
      >
        <Stack gap="density-md">
          <Text>Cancelling stops this evaluation. Partial results are not retained.</Text>
        </Stack>
      </Modal>
    </AccessibleTitle>
  );
};
