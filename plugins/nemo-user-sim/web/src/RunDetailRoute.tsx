// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  DeleteConfirmationModal,
  StatusBadge,
  StudioDataView,
  TableEmptyState,
  useStudioDataViewState,
} from '@nemo/common';
import { Button, Card, Flex, Stack, Tag, Text } from '@nvidia/foundations-react-core';
import { useEffect, useMemo, useState, type ComponentProps } from 'react';
import { useNavigate, useParams } from 'react-router';
import {
  isFinished,
  useCancelRuns,
  useCreateJob,
  useDeleteRuns,
  useProbes,
  useRunResults,
  useSimulateJob,
} from './api';
import { ExportDialog } from './ExportDialog';
import { useHost } from './host';
import { getRunListRoute, getRunRoute } from './paths';
import { ConversationViewer } from './ConversationViewer';
import { NewRunPanel } from './NewRunPanel';
import type { SimulateRequest } from './types';
import {
  axisLabel,
  otherFailures,
  pickText,
  failureRates,
  joinScores,
  SEGMENTS,
  styleColor,
  words,
  type Row,
} from './report';

/**
 * A conversation's outcome. A scenario set's own pick label replaces the tool check's wording, and a run
 * error reads "Failed to run", so neither can be mistaken for the other.
 */
const outcome = (row: Row): string => {
  if (row.status === 'running') return 'Running';
  if (row.status === 'failed')
    return `Failed to run: ${row.error?.message ?? 'no reason recorded'}`;
  if (row.passed === null) return 'Not scored';
  const others = otherFailures(row).map(axisLabel);
  if (row.picked) return [pickText(row.picked), ...others].join('; also ');
  if (row.passed) return 'Passed';
  return others.length ? `Failed: ${others.join('; ')}` : 'Failed scoring';
};

/** The scenario set's label for the tool it shouldn't call, when it has one and any conversation called it. */
const wrongPickLabel = (rows: Row[]) =>
  rows.find((row) => row.picked?.pick === 'should_not_call')?.picked?.label ?? null;

const Stat = ({ label, value }: { label: string; value: string | number }) => (
  <Card className="flex-1 p-4">
    <Stack gap="density-xs">
      <Text kind="body/regular/sm" className="text-subtle">
        {label}
      </Text>
      <Text kind="body/semibold/xl">{value}</Text>
    </Stack>
  </Card>
);

// The grid is inline because Studio compiles only the Tailwind classes its own sources use.
const SegmentBreakdown = ({
  rows,
  probeLabel,
  byWrongPick,
}: {
  rows: Row[];
  probeLabel: (name: string) => string;
  /** Count the scenario set's wrong pick rather than every failure. */
  byWrongPick: boolean;
}) => (
  <div
    style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
      gap: 12,
    }}
  >
    {SEGMENTS.map(([segment, label]) => (
      <Card key={segment} className="p-4">
        <Stack gap="density-xs">
          <Text kind="body/semibold/sm">{label}</Text>
          {failureRates(
            rows,
            segment,
            byWrongPick ? (row) => row.picked?.pick === 'should_not_call' : undefined
          ).map((rate) => (
            <Flex key={rate.value} justify="between" gap="density-md">
              <Text kind="body/regular/sm" className="truncate">
                {segment === 'probe_type' ? probeLabel(rate.value) : words(rate.value)}
              </Text>
              <Text kind="body/regular/sm" className="whitespace-nowrap text-subtle">
                {rate.failed} of {rate.scored}
                {byWrongPick ? '' : ' failed'}
              </Text>
            </Flex>
          ))}
        </Stack>
      </Card>
    ))}
  </div>
);

export const RunDetailRoute = () => {
  const { job = '' } = useParams();
  const { workspaceId, breadcrumbs, notifications } = useHost();
  const { data: run, isError } = useSimulateJob(job);
  const finished = isFinished(run?.status);
  const { summary, scores, transcripts, scoringError } = useRunResults(job, finished);
  const rows = useMemo(
    () => joinScores(summary.data?.conversations ?? [], scores.data ?? null),
    [summary.data, scores.data]
  );
  const dataViewState = useStudioDataViewState();
  const [exporting, setExporting] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [rerunning, setRerunning] = useState(false);
  const navigate = useNavigate();
  const deleteRuns = useDeleteRuns();
  const cancelRuns = useCancelRuns();
  const rerun = useCreateJob<SimulateRequest>('simulate');
  const [viewing, setViewing] = useState<number | null>(null);
  const { data: catalog = [] } = useProbes();
  const probes = useMemo(() => new Map(catalog.map((probe) => [probe.name, probe])), [catalog]);
  const probeLabel = (name: string) => probes.get(name)?.label ?? words(name);

  useEffect(() => {
    breadcrumbs.set([
      { label: 'User Simulation', href: getRunListRoute(workspaceId) },
      { label: job },
    ]);
    return () => breadcrumbs.set([]);
  }, [breadcrumbs, workspaceId, job]);

  const makeColumns = useMemo<ComponentProps<typeof StudioDataView<Row>>['makeColumns']>(
    () => (col, { rowSelectionColumn }) => [
      rowSelectionColumn({ size: 48 }),
      col.accessor('persona_name', { header: 'Persona', size: 180 }),
      col.accessor('persona_occupation', {
        header: 'Occupation',
        size: 180,
        cell: ({ getValue }) => words(getValue()),
      }),
      col.accessor('user_interaction_style', {
        header: 'Style',
        size: 160,
        cell: ({ getValue }) => (
          <Tag kind="solid" color={styleColor(getValue())} density="compact" readOnly>
            {getValue()}
          </Tag>
        ),
      }),
      col.accessor('probe_type', {
        header: 'Probe',
        size: 180,
        cell: ({ getValue }) => probes.get(getValue() ?? '')?.label ?? words(getValue()),
      }),
      col.display({
        id: 'outcome',
        header: 'Outcome',
        size: 320,
        cell: ({ row }) => outcome(row.original),
      }),
    ],
    [probes]
  );

  if (isError)
    return <TableEmptyState header="Run not found" emptyMessage={`No simulation named ${job}.`} />;

  const passed = rows.filter((row) => row.passed).length;
  const scored = rows.filter((row) => row.passed !== null).length;
  const totals = summary.data?.totals;
  const target = summary.data?.target ?? run?.spec.target;
  const model = summary.data?.assistant_model ?? target?.model;
  const tested = target?.agent ?? target?.model ?? 'The agent';
  const wrongPick = wrongPickLabel(rows);

  return (
    <Stack gap="density-lg" className="p-4">
      <Flex justify="between" align="center">
        <Stack gap="density-xs">
          <Flex align="center" gap="density-md">
            <Text kind="body/semibold/lg">{job}</Text>
            <StatusBadge status={run?.status} />
          </Flex>
          {totals ? (
            <Text kind="body/regular/sm" className="text-subtle">
              {tested}
              {target?.agent && model ? ` (${model})` : ''} talked to {totals.conversations}{' '}
              simulated {totals.conversations === 1 ? 'user' : 'users'}.
            </Text>
          ) : null}
          {target?.use_agent_tools ? (
            <Text kind="body/regular/sm" className="text-subtle">
              Agent&apos;s own tools: on
            </Text>
          ) : null}
        </Stack>
        <Flex gap="density-sm">
          {run && !finished ? (
            <Button
              kind="tertiary"
              onClick={() =>
                cancelRuns.mutate([job], {
                  onSuccess: () => notifications.notify(`Cancelling ${job}.`, 'success'),
                  onError: (error) => notifications.notify(error.message, 'error'),
                })
              }
            >
              Cancel run
            </Button>
          ) : null}
          <Button kind="tertiary" color="danger" onClick={() => setDeleting(true)}>
            Delete
          </Button>
          {run ? (
            <Button kind="secondary" onClick={() => setRerunning(true)}>
              Run again
            </Button>
          ) : null}
          <Button kind="secondary" disabled={passed === 0} onClick={() => setExporting(true)}>
            Export training data
          </Button>
        </Flex>
      </Flex>
      {scoringError.data?.message || run?.error_details?.message ? (
        <Text kind="body/regular/sm" color="danger">
          {scoringError.data?.message ?? run?.error_details?.message}
        </Text>
      ) : null}
      {summary.isError ? (
        <TableEmptyState header="Couldn't load results" emptyMessage={summary.error.message} />
      ) : summary.isPending ? (
        <Text kind="body/regular/sm" className="text-subtle">
          Loading results…
        </Text>
      ) : totals ? (
        <>
          <Flex gap="density-md">
            <Stat label="Conversations" value={totals.conversations} />
            {totals.running ? <Stat label="Running" value={totals.running} /> : null}
            <Stat
              label="Passed scoring"
              value={scored ? `${passed} of ${scored}` : 'Not scored yet'}
            />
            <Stat label="Failed to run" value={totals.failed} />
          </Flex>
          {scored > passed ? (
            <Stack gap="density-sm">
              <Text kind="body/semibold/md">
                {wrongPick
                  ? `Where it ${wrongPick.charAt(0).toLowerCase()}${wrongPick.slice(1)}`
                  : 'Where it fails'}
              </Text>
              <SegmentBreakdown rows={rows} probeLabel={probeLabel} byWrongPick={!!wrongPick} />
            </Stack>
          ) : scored ? (
            <Text kind="body/regular/sm" className="text-subtle">
              Every scored conversation passed.
            </Text>
          ) : null}
          <Text kind="body/regular/sm" className="text-subtle">
            Select a conversation to read it. Conversations appear here as they finish.
          </Text>
          <StudioDataView<Row>
            dataViewState={dataViewState}
            makeColumns={makeColumns}
            onRowClick={(row) => setViewing(rows.indexOf(row))}
            renderBulkActions={({ selectedRows, table }) => {
              // A conversation still running has nothing to rerun yet.
              const done = selectedRows.filter((row) => row.status !== 'running');
              if (!run || !done.length) return null;
              const { target, probes: runProbes, locale, max_turns, scenarios } = run.spec;
              const spec: SimulateRequest = {
                target,
                probes: runProbes,
                locale,
                users: done.length,
                ...(max_turns ? { max_turns } : {}),
                ...(scenarios ? { scenarios } : {}),
                rerun: { job, conversations: done.map((row) => row.trajectory_id) },
              };
              return (
                <Button
                  kind="tertiary"
                  disabled={rerun.isPending}
                  onClick={() =>
                    rerun.mutate(spec, {
                      onSuccess: (created) => {
                        table.resetRowSelection();
                        notifications.notify(`Rerunning as ${created.name}.`, 'success');
                        navigate(getRunRoute(workspaceId, created.name));
                      },
                      onError: (error) => notifications.notify(error.message, 'error'),
                    })
                  }
                >
                  Rerun {done.length} {done.length === 1 ? 'conversation' : 'conversations'}
                </Button>
              );
            }}
            attributes={{
              DataViewRoot: {
                data: rows,
                totalCount: rows.length,
                reactTableOptions: { getRowId: (row) => row.trajectory_id },
              },
            }}
          />
          <ConversationViewer
            rows={rows}
            index={viewing}
            transcripts={transcripts.data}
            outcome={outcome}
            agent={tested}
            model={model}
            probes={probes}
            onIndex={setViewing}
            onClose={() => setViewing(null)}
          />
        </>
      ) : (
        <TableEmptyState
          header={finished ? 'No results' : 'Starting'}
          emptyMessage={
            finished
              ? 'This run ended before any conversation started. Check the job logs.'
              : 'Sampling personas. Conversations appear here as soon as they start.'
          }
        />
      )}
      {run ? (
        // Keyed so each opening starts from the run's settings, not the last edits.
        <NewRunPanel
          key={String(rerunning)}
          open={rerunning}
          onClose={() => setRerunning(false)}
          from={{ job, spec: run.spec }}
        />
      ) : null}
      <ExportDialog jobs={[job]} open={exporting} onClose={() => setExporting(false)} />
      <DeleteConfirmationModal
        open={deleting}
        onClose={() => setDeleting(false)}
        onNotify={notifications.notify}
        simpleConfirm
        title={`Delete ${job}?`}
        description="The run and its results are deleted. Its conversations stay in Traces, and training data you already exported is kept."
        successText="Run deleted."
        errorText="The run couldn't be deleted."
        onDelete={async () => {
          await deleteRuns.mutateAsync([job]);
          navigate(getRunListRoute(workspaceId));
          return true;
        }}
      />
    </Stack>
  );
};
