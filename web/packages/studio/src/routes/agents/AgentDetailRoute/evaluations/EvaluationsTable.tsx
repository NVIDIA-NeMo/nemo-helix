// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ROW_SELECTION_COLUMN_SIZE,
  StudioDataView,
} from '@nemo/common/src/components/DataView/StudioDataView';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { TableEmptyState } from '@nemo/common/src/components/TableEmptyState';
import { PlatformJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { useLiveSeconds } from '@nemo/common/src/hooks/useLiveSeconds';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { formatDurationMs, formatTimeInSeconds, utcToLocalDate } from '@nemo/common/src/utils/date';
import {
  deleteEvaluation,
  getListEvaluationsQueryKey,
} from '@nemo/sdk/generated/platform/evaluations';
import { Button, Flex, Text } from '@nvidia/foundations-react-core';
import { type EvalJobRow, evalDurationMs } from '@studio/api/evaluation/utils';
import { BulkDeleteModal } from '@studio/components/BulkDeleteModal';
import { evaluationFilesetName } from '@studio/components/evaluation/experimentEvalConfig';
import { SubmitEvaluationModal } from '@studio/components/evaluation/SubmitEvaluationModal';
import { evaluatorScores } from '@studio/routes/agents/AgentDetailRoute/evaluations/formatRollups';
import {
  type AgentEvaluationRow,
  primaryExperimentName,
} from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';
import { getEvaluationDetailRoute, getWorkspaceJobDetailRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { FlaskConical, Trash } from 'lucide-react';
import { type ComponentProps, type FC, useCallback, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router';

interface AgentEvalTableRow {
  id: string;
  evaluation: AgentEvaluationRow | null;
  job: EvalJobRow | null;
}

const isTerminalJob = (job: EvalJobRow): boolean =>
  PlatformJobTerminalStatuses.some((status) => status === job.status);

const timestampMs = (value?: string): number => {
  const time = utcToLocalDate(value)?.getTime();
  return time != null && Number.isFinite(time) ? time : 0;
};

const createdAtMs = (row: AgentEvalTableRow): number =>
  timestampMs(row.job?.created_at ?? row.evaluation?.created_at);

const preferredJob = (a: EvalJobRow, b: EvalJobRow): EvalJobRow => {
  if (isTerminalJob(a) !== isTerminalJob(b)) return isTerminalJob(a) ? b : a;
  return timestampMs(b.created_at) >= timestampMs(a.created_at) ? b : a;
};

const mergeRows = (evaluations: AgentEvaluationRow[], jobs: EvalJobRow[]): AgentEvalTableRow[] => {
  const jobByEvaluation: Record<string, EvalJobRow> = {};
  for (const job of jobs) {
    if (!job.evaluationName) continue;
    const claimed = jobByEvaluation[job.evaluationName];
    jobByEvaluation[job.evaluationName] = claimed ? preferredJob(claimed, job) : job;
  }

  const published = new Set(evaluations.map((evaluation) => evaluation.name));
  const rows: AgentEvalTableRow[] = evaluations.map((evaluation) => ({
    id: `eval:${evaluation.name}`,
    evaluation,
    job: jobByEvaluation[evaluation.name] ?? null,
  }));
  for (const job of jobs) {
    if (job.evaluationName && published.has(job.evaluationName)) continue;
    rows.push({ id: `job:${job.id}`, evaluation: null, job });
  }

  return rows.sort((a, b) => createdAtMs(b) - createdAtMs(a));
};

/** Elapsed time for one row: a live counter while its job runs, the recorded duration once it does
 *  not.
 *
 *  A completed job's own `updated_at` is not an end time — the job row is written at create and on
 *  rerun only, never on a status transition — so the duration has to come from the evaluation. */
const DurationCell: FC<{ job: EvalJobRow | null; durationMs?: number }> = ({ job, durationMs }) => {
  const runningJob = job && !isTerminalJob(job) ? job : null;
  // `enabled` is what actually stops the timer: the hook's interval effect keys off its *locked*
  // start date, so clearing `startDate` alone leaves a row that finished mid-poll ticking (and
  // re-rendering the table) once a second forever.
  const liveSeconds = useLiveSeconds({
    startDate: utcToLocalDate(runningJob?.created_at),
    enabled: runningJob != null,
  });
  if (runningJob) return <Text>{formatTimeInSeconds(liveSeconds) || '0s'}</Text>;
  return <Text>{formatDurationMs(durationMs, { hideMsAboveMinute: true })}</Text>;
};

interface EvaluationsTableProps {
  workspace: string;
  /** The agent these evaluations belong to, seeded into a re-run started from a row. */
  agentName?: string;
  evaluations: AgentEvaluationRow[];
  /** All evaluator jobs for the agent (any status). Supplies the status, live duration and job link
   *  for a published evaluation, and stands in as a whole row for a run that has yet to publish. */
  jobs: EvalJobRow[];
}

/** Every evaluation for the agent, ungrouped, from the moment its run is submitted. */
export const EvaluationsTable: FC<EvaluationsTableProps> = ({
  workspace,
  agentName,
  evaluations,
  jobs,
}) => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const dataViewState = useStudioDataViewState();
  const [deleteRows, setDeleteRows] = useState<AgentEvaluationRow[]>([]);
  // The evaluation a "New evaluation from this configuration" click is re-running, or null when
  // the form is closed.
  const [rerunSource, setRerunSource] = useState<AgentEvaluationRow | null>(null);
  const rows = useMemo(() => mergeRows(evaluations, jobs), [evaluations, jobs]);
  // StudioDataView is `dataMode="manual"`, so it renders exactly the rows it is handed and leaves
  // paging to the caller. Both source lists are already in memory, so slice here rather than
  // refetching a page.
  const { pageIndex, pageSize } = dataViewState.pagination.state;
  const pageRows = useMemo(
    () => rows.slice(pageIndex * pageSize, pageIndex * pageSize + pageSize),
    [rows, pageIndex, pageSize]
  );

  const handleDelete = useCallback(
    async (rowsToDelete: AgentEvaluationRow[]) => {
      const results = await Promise.allSettled(
        rowsToDelete.map((row) => deleteEvaluation(workspace, row.name))
      );
      await queryClient.invalidateQueries({ queryKey: getListEvaluationsQueryKey(workspace) });

      const failed = rowsToDelete.filter((_, index) => results[index]?.status === 'rejected');
      if (failed.length) {
        setDeleteRows(failed);
        throw new Error(
          `${failed.length} of ${rowsToDelete.length} evaluation${rowsToDelete.length !== 1 ? 's' : ''} could not be deleted. Retry to attempt only those.`
        );
      }
    },
    [workspace, queryClient]
  );

  // Always the evaluation's own results, never the job that produced them — the Job cell links
  // there. The route is nested under an experiment, so a row whose experiment has not resolved has
  // nowhere to go.
  const destinationFor = useCallback(
    (row: AgentEvalTableRow): string | undefined => {
      const experimentName = row.evaluation ? primaryExperimentName(row.evaluation) : undefined;
      return row.evaluation && experimentName
        ? getEvaluationDetailRoute(workspace, experimentName, row.evaluation.name)
        : undefined;
    },
    [workspace]
  );

  const makeColumns: ComponentProps<typeof StudioDataView<AgentEvalTableRow>>['makeColumns'] =
    useCallback(
      ({ accessor }, { rowSelectionColumn, rowActionsColumn }) => [
        rowSelectionColumn({ size: ROW_SELECTION_COLUMN_SIZE }),
        accessor((row) => row.evaluation?.name ?? row.job?.evaluationName ?? undefined, {
          id: 'name',
          header: 'Evaluation',
          cell: ({ row }) => {
            const name = row.original.evaluation?.name ?? row.original.job?.evaluationName;
            return <Text title={name ?? undefined}>{name ?? '—'}</Text>;
          },
        }),
        accessor((row) => (row.evaluation ? primaryExperimentName(row.evaluation) : undefined), {
          id: 'experiment',
          header: 'Experiment',
          size: 200,
          cell: ({ row }) => {
            const { evaluation } = row.original;
            const name = evaluation ? primaryExperimentName(evaluation) : undefined;
            return <Text title={name}>{name ?? '—'}</Text>;
          },
        }),
        accessor((row) => row.job?.status ?? (row.evaluation ? 'completed' : undefined), {
          id: 'status',
          header: 'Status',
          cell: ({ row }) => {
            const { job, evaluation } = row.original;
            if (job) return <StatusBadge status={job.status} />;
            return evaluation ? <StatusBadge status="completed" /> : <Text>—</Text>;
          },
        }),
        accessor((row) => row.evaluation?.test_case_count, {
          id: 'test_case_count',
          header: 'Test Cases',
          size: 100,
          cell: ({ row }) => {
            const count = row.original.evaluation?.test_case_count;
            return <Text>{count ? count : '—'}</Text>;
          },
        }),
        accessor((row) => row.evaluation?.aggregate_scores, {
          id: 'aggregate_scores',
          header: 'Scores',
          enableSorting: false,
          cell: ({ row }) => {
            const { evaluation } = row.original;
            const scores = evaluation ? evaluatorScores(evaluation) : [];
            if (scores.length === 0) return <Text>—</Text>;
            return (
              <Flex gap="density-xs" className="flex-wrap">
                {scores.map((score) => (
                  <Flex
                    key={score.key}
                    gap="density-xxs"
                    align="baseline"
                    className="rounded bg-surface-raised px-1.5 py-0.5"
                  >
                    <Text kind="body/regular/sm" color="secondary">
                      {score.label}
                    </Text>
                    <Text kind="body/semibold/sm">{score.value}</Text>
                  </Flex>
                ))}
              </Flex>
            );
          },
        }),
        accessor((row) => row.evaluation?.tokens?.mean, {
          id: 'tokens',
          header: 'Avg Tokens',
          enableSorting: false,
          cell: ({ row }) => {
            const mean = row.original.evaluation?.tokens?.mean;
            return <Text>{mean != null ? Math.round(mean).toLocaleString() : '—'}</Text>;
          },
        }),
        accessor((row) => row.evaluation?.tokens?.sum, {
          id: 'total_tokens',
          header: 'Total Tokens',
          enableSorting: false,
          cell: ({ row }) => {
            const sum = row.original.evaluation?.tokens?.sum;
            return <Text>{sum != null ? Math.round(sum).toLocaleString() : '—'}</Text>;
          },
        }),
        accessor((row) => evalDurationMs(row.evaluation?.metadata), {
          id: 'duration',
          header: 'Duration',
          enableSorting: false,
          // Keyed by job: an evaluation row outlives the job it shows, and `useLiveSeconds` locks
          // its start date once, so a rerun would otherwise tick from the previous job's start.
          cell: ({ row }) => (
            <DurationCell
              key={row.original.job?.id}
              job={row.original.job}
              durationMs={evalDurationMs(row.original.evaluation?.metadata)}
            />
          ),
        }),
        accessor(createdAtMs, {
          id: 'created_at',
          header: 'Created',
          cell: ({ row }) => {
            const datetime = row.original.job?.created_at ?? row.original.evaluation?.created_at;
            return datetime ? <RelativeTime datetime={datetime} /> : '—';
          },
        }),
        accessor((row) => row.job?.name, {
          id: 'job',
          header: 'Job',
          cell: ({ row }) => {
            const { job } = row.original;
            if (!job) return <Text>—</Text>;
            return (
              <Link
                to={getWorkspaceJobDetailRoute(workspace, job.name)}
                className="text-primary underline"
                title={job.name}
              >
                {job.name}
              </Link>
            );
          },
        }),
        rowActionsColumn({
          rowActions: (row) => {
            const { evaluation, job } = row;
            const actions = [
              // Only offered for a row that carries a reusable eval config; without one the form
              // would open on an evaluation it has to reject.
              ...(evaluation && evaluationFilesetName(evaluation)
                ? [
                    {
                      children: 'New evaluation from this configuration',
                      onSelect: () => setRerunSource(evaluation),
                    },
                  ]
                : []),
              ...(job
                ? [
                    {
                      children: 'View job',
                      onSelect: () => navigate(getWorkspaceJobDetailRoute(workspace, job.name)),
                    },
                  ]
                : []),
            ];
            return actions.length > 0 ? actions : false;
          },
        }),
      ],
      [navigate, workspace]
    );

  return (
    <>
      <StudioDataView<AgentEvalTableRow>
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        onRowClick={(row) => {
          const destination = destinationFor(row);
          if (destination) navigate(destination);
        }}
        renderBulkActions={({ selectedRows }) => (
          <Button
            kind="tertiary"
            aria-label="Delete selected evaluations"
            onClick={() =>
              setDeleteRows(selectedRows.flatMap((row) => (row.evaluation ? [row.evaluation] : [])))
            }
          >
            <Trash /> Delete
          </Button>
        )}
        attributes={{
          DataViewRoot: {
            data: pageRows,
            totalCount: rows.length,
            reactTableOptions: {
              enableRowSelection: (row) => row.original.evaluation != null,
            },
          },
          DataViewTableContent: {
            renderEmptyState: () => (
              <TableEmptyState
                className="py-density-3xl"
                icon={<FlaskConical className="size-16" />}
                header="No evaluations yet"
                emptyMessage="Runs appear here as soon as they are submitted, before any results are published."
              />
            ),
          },
        }}
      />

      {rerunSource && (
        <SubmitEvaluationModal
          open
          onClose={() => setRerunSource(null)}
          workspace={workspace}
          agent={agentName ?? rerunSource.agent_names?.[0]}
          sourceEvaluation={rerunSource.name}
        />
      )}

      <BulkDeleteModal
        items={deleteRows}
        open={deleteRows.length > 0}
        onDelete={handleDelete}
        title={(count) => `Delete ${count} Evaluation${count !== 1 ? 's' : ''}`}
        onClose={() => {
          setDeleteRows([]);
          dataViewState.rowSelection.set({});
        }}
      />
    </>
  );
};
