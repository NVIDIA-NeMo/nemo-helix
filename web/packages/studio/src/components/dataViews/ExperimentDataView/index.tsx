// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { dateTimeFilter } from '@nemo/common/src/components/DataView/dateTimeFilter';
import { numberRangeFilter } from '@nemo/common/src/components/DataView/FilterPanel/NumberRangeFilter/util';
import {
  Root as DataViewRoot,
  EditColumnsMenu,
} from '@nemo/common/src/components/DataView/internal';
import {
  ROW_ACTIONS_COLUMN_SIZE,
  ROW_SELECTION_COLUMN_SIZE,
  StudioDataView,
} from '@nemo/common/src/components/DataView/StudioDataView';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { QuickActionsMenuRoot } from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { formatDurationMs } from '@nemo/common/src/utils/date';
import { formatEvaluatorScore, snakeCaseToTitleCase } from '@nemo/common/src/utils/formatters';
import type { EvaluationFilter, ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { Button, Text, Tooltip } from '@nvidia/foundations-react-core';
import { EVAL_DURATION_METADATA_KEY, evalDurationMs } from '@studio/api/evaluation/utils';
import { ChangesetBadge } from '@studio/components/ChangesetBadge';
import { ExperimentParetoChart } from '@studio/components/charts/ExperimentParetoChart';
import { ExperimentTrendChart } from '@studio/components/charts/ExperimentTrendChart';
import { AddToGroupModal } from '@studio/components/dataViews/ExperimentDataView/AddToGroupModal';
import '@studio/components/dataViews/ExperimentDataView/ExperimentDataView.css';
import { MeanValueTooltipCell } from '@studio/components/dataViews/ExperimentDataView/MeanValueTooltipCell';
import {
  seedColumnState,
  useColumnLayout,
} from '@studio/components/dataViews/ExperimentDataView/useColumnLayout';
import {
  type EvaluationRow,
  type ListEvaluationsSortParam,
  useExperimentEvaluations,
} from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';
import { useSortErrorRecovery } from '@studio/components/dataViews/ExperimentDataView/useSortErrorRecovery';
import { deriveEvaluatorNames } from '@studio/components/dataViews/ExperimentDataView/util';
import { evaluationFilesetName } from '@studio/components/evaluation/experimentEvalConfig';
import { SubmitEvaluationModal } from '@studio/components/evaluation/SubmitEvaluationModal';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getEvaluationDetailRoute } from '@studio/routes/utils';
import { tooltipClassName } from '@studio/styles/common';
import { Columns3, FolderMinus, FolderPlus, Pin, Repeat2 } from 'lucide-react';
import { type ComponentProps, type FC, useCallback, useMemo, useState } from 'react';

export type { EvaluationRow };

const DEFAULT_SORT: ListEvaluationsSortParam = '-created_at';

// Maps sortable static column ids to their API sort fields.
const STATIC_SORT_FIELD_MAP: Readonly<Record<string, string>> = {
  name: 'name',
  created_at: 'created_at',
  cost_usd: 'cost_usd.mean',
  latency_ms: 'latency_ms.mean',
  total_latency_ms: 'latency_ms.sum',
  tokens: 'tokens.mean',
  total_tokens: 'tokens.sum',
  test_case_count: 'test_case_count',
};

// Resolves a group's default_sort (a `sort`-param string like `-cost_usd.mean` or `-created_at`) to
// the table's initial sort so the matching column header shows the sort on load. Entity columns map
// by exact id; metric columns match on the family (any stat) since the column always sorts on `.mean`.
// Maps one API sort field to its table column id: entity columns by exact id; metric columns by
// family (any stat) since the column always sorts on `.mean`.
const sortFieldToColumnId = (field: string): string | undefined => {
  if (field === 'name' || field === 'created_at' || field === 'test_case_count') return field;
  if (field.startsWith('cost_usd.')) return 'cost_usd';
  if (field === 'latency_ms.sum') return 'total_latency_ms';
  if (field.startsWith('latency_ms.')) return 'latency_ms';
  if (field === 'tokens.sum') return 'total_tokens';
  if (field.startsWith('tokens.')) return 'tokens';
  const evaluatorMatch = field.match(/^evaluators\.(.+)\.[^.]+$/);
  if (evaluatorMatch) return `evaluator-${evaluatorMatch[1]}`;
  return undefined;
};

// Seeds the table's (multi-column) initial sort from the group's default_sort — a comma-separated,
// ordered list of API sort fields — so the column headers reflect the saved order on load.
const seedSortFromDefault = (
  defaultSort: string | null | undefined
): { id: string; desc: boolean }[] | undefined => {
  if (!defaultSort) return undefined;
  const entries = defaultSort
    .split(',')
    .map((token) => token.trim())
    .filter(Boolean)
    .map((token) => {
      const desc = token.startsWith('-');
      const id = sortFieldToColumnId(desc ? token.slice(1) : token);
      return id ? { id, desc } : undefined;
    })
    .filter((entry): entry is { id: string; desc: boolean } => entry !== undefined);
  return entries.length ? entries : undefined;
};

// Maps a filter column id to its dotted API rollup-stat field (required by the backend parser).
// Evaluator ids are dynamic, so derive `evaluators.<name>.mean` here, like getEvaluationSortParam.
const getEvaluationFilterField = (id: string): string | undefined => {
  if (id === 'cost_usd') return 'cost_usd.mean';
  if (id === 'latency_ms') return 'latency_ms.mean';
  if (id === 'total_latency_ms') return 'latency_ms.sum';
  if (id === 'tokens') return 'tokens.mean';
  if (id === 'total_tokens') return 'tokens.sum';
  // The list columns hold the plural name facets; the API filter params are singular contains-matches.
  if (id === 'agent_names') return 'agent_name';
  if (id === 'agent_versions') return 'agent_version';
  if (id === 'model_names') return 'model_name';
  const evaluatorMatch = id.match(/^evaluator-(.+)$/);
  if (evaluatorMatch) return `evaluators.${evaluatorMatch[1]}.mean`;
  return undefined;
};

// Resolves one sorting-state entry to its API sort field (with '-' prefix for descending), or
// undefined when the column has no corresponding API sort field.
const resolveEvaluationSortField = (entry: { id: string; desc: boolean }): string | undefined => {
  let field = STATIC_SORT_FIELD_MAP[entry.id];
  if (!field) {
    // Evaluator columns use id `evaluator-<name>` so the API field can be derived without
    // a separate lookup into evaluatorNames (which would create a circular dependency).
    const evaluatorMatch = entry.id.match(/^evaluator-(.+)$/);
    if (evaluatorMatch) field = `evaluators.${evaluatorMatch[1]}.mean`;
  }
  if (!field) return undefined;
  return `${entry.desc ? '-' : ''}${field}`;
};

// Emits the API `sort` param from the table's (multi-column) sorting state: a comma-separated,
// ordered list of fields — the first sorted column dominates, matching the API's key precedence.
const getEvaluationSortParam = (
  sortingState: { id: string; desc: boolean }[]
): ListEvaluationsSortParam | undefined => {
  // No column sort -> omit `sort`; the API then defaults to -created_at with pinned first.
  if (sortingState.length === 0) return undefined;
  const fields = sortingState
    .map(resolveEvaluationSortField)
    .filter((field): field is string => field !== undefined);
  if (fields.length === 0) return DEFAULT_SORT;
  return fields.join(',') as ListEvaluationsSortParam;
};

interface ExperimentDataViewProps {
  /** The loaded group, so the table's initial sort can seed from `default_sort` at first
   * render — the sorting state is initialized once and not reactive. */
  group: ExperimentResponse;
  /** Whether the Pareto (cost-vs-accuracy) chart is shown. The toggle lives in the parent
   * route header (next to the "Evaluations" title); this component only renders the chart. */
  paretoVisible: boolean;
  /** Whether the over-time trend chart is shown. Same arrangement as `paretoVisible`: the parent
   * owns the toggle, which starts on for experiments flagged `show_evaluations_over_time`. */
  trendVisible: boolean;
}

/** Lists the experiments that belong to a single experiment group. */
export const ExperimentDataView: FC<ExperimentDataViewProps> = ({
  group,
  paretoVisible,
  trendVisible,
}) => {
  const workspace = useWorkspaceFromPath();
  const openRow = useRowNavigation();
  const toast = useToast();
  const experimentName = group.name;
  const experimentId = group.id;

  // The bulk "Add to group" modal state: the selected evaluations plus a way to clear the row
  // selection on success. Null when the modal is closed.
  const [addToGroupState, setAddToGroupState] = useState<{
    evaluations: EvaluationRow[];
    clearSelection: () => void;
  } | null>(null);

  // The evaluation a "New evaluation from this configuration" click is re-running, or null when
  // the form is closed. Re-running is reachable from here as well as from the agent page, because
  // this leaderboard is where you notice a run worth varying.
  const [rerunSource, setRerunSource] = useState<EvaluationRow | null>(null);

  // Seed the sort from default_sort so its column header reflects the order on load. Memoized so the
  // reference is stable across renders (until default_sort changes).
  const defaultSort = useMemo(() => seedSortFromDefault(group.default_sort), [group.default_sort]);

  const seededColumns = useMemo(() => seedColumnState(group.column_layout), [group.column_layout]);

  const dataViewState = useStudioDataViewState<EvaluationFilter>({
    defaultSort,
    // The evaluations leaderboard supports multi-column sort (shift-click) — score vs. cost etc.
    multiSort: true,
    columnVisibility: seededColumns.columnVisibility,
    // Keep the pin toggle, row selection, and the row actions menu reachable while horizontally
    // scrolling this wide table.
    columnPinning: { left: ['pin', 'row-selection'], right: ['row-actions'] },
    filterFieldMap: getEvaluationFilterField,
    columnOrder: seededColumns.columnOrder,
  });

  const columnLayout = useColumnLayout({
    workspace,
    group,
    columnOrder: dataViewState.columnOrder.state,
    columnVisibility: dataViewState.columnVisibility.state,
  });

  const page = dataViewState.pagination.state.pageIndex + 1;
  const pageSize = dataViewState.pagination.state.pageSize;
  const sortParam = getEvaluationSortParam(dataViewState.sorting.state);

  const {
    rows: orderedData,
    togglePin,
    removeFromGroup,
    totalCount,
    error,
    isLoading,
    isSuccess,
  } = useExperimentEvaluations({
    workspace,
    experimentId,
    filter: dataViewState.apiFilter.filter,
    search: dataViewState.debouncedSearchBar,
    page,
    pageSize,
    sort: sortParam,
  });

  // A metric sort (cost, latency, evaluator score) can fail with 413/503/400. Recover by toasting
  // and reverting the sort indicator to the last good sort instead of replacing the table with an
  // error; `isRecoverableSortError` lets us skip the page-level error for that case.
  const isRecoverableSortError = useSortErrorRecovery({
    error,
    isSuccess,
    sortingState: dataViewState.sorting.state,
    setSorting: dataViewState.sorting.set,
    onError: toast.error,
  });

  // One score column per evaluator: the names found across the loaded rows, plus any evaluator
  // with an active filter (so its column survives a zero-result filter — see deriveEvaluatorNames).
  const evaluatorNames = useMemo(
    () => deriveEvaluatorNames(orderedData, dataViewState.debouncedColumnFilters),
    [orderedData, dataViewState.debouncedColumnFilters]
  );

  // One column per metadata key: keys are lowercased so case variants (e.g. "status"
  // and "Status") collapse into one column rather than producing duplicate headers.
  // The run duration is excluded because the Duration column below already renders it, formatted;
  // publish duration is deliberately left in, which is what surfaces publish latency at all.
  const metadataKeys = useMemo(
    () =>
      [
        ...new Set(
          orderedData.flatMap((e) => Object.keys(e.metadata ?? {}).map((k) => k.toLowerCase()))
        ),
      ]
        .filter((key) => key !== EVAL_DURATION_METADATA_KEY)
        .sort(),
    [orderedData]
  );

  const makeColumns = useCallback<
    ComponentProps<typeof DataViewRoot<EvaluationRow>>['makeColumns']
  >(
    ({ accessor, display }, { rowSelectionColumn, rowActionsColumn }) => [
      rowSelectionColumn({ size: ROW_SELECTION_COLUMN_SIZE }),
      display({
        id: 'pin',
        header: () => <span className="sr-only">Pinned</span>,
        enableSorting: false,
        enableHiding: false,
        enableResizing: false,
        size: 40,
        minSize: 40,
        maxSize: 40,
        meta: { alignment: 'center', _isPrebuiltColumn: true, _isSizeInitialized: true },
        cell: ({ row }) => {
          const { pinned_at } = row.original;
          const isPinned = pinned_at != null;
          return (
            <Tooltip
              slotContent={
                <Text kind="body/regular/sm">
                  {isPinned ? 'Unpin for all users' : 'Pin for all users'}
                </Text>
              }
              className={tooltipClassName}
              side="right"
            >
              <Button
                kind="tertiary"
                color="neutral"
                size="small"
                aria-label={isPinned ? 'Unpin evaluation' : 'Pin evaluation'}
                aria-pressed={isPinned}
                onClick={() => togglePin(row.original)}
              >
                <Pin
                  className={isPinned ? 'text-brand' : 'text-secondary'}
                  {...(isPinned ? { fill: 'currentColor' } : {})}
                />
              </Button>
            </Tooltip>
          );
        },
      }),
      accessor('name', {
        header: 'Name',
        enableSorting: true,
        enableHiding: false,
        meta: { title: false },
        size: 300,
        cell: ({ row }) => {
          const { name, description } = row.original;
          if (!description) return <Text>{name}</Text>;
          return (
            <Tooltip
              slotContent={
                <div className="flex flex-col gap-1">
                  <Text kind="label/regular/sm" className="text-secondary">
                    Description
                  </Text>
                  <Text kind="body/regular/sm">{description}</Text>
                </div>
              }
              className={tooltipClassName}
              side="bottom"
            >
              <Text className="cursor-default border-b border-dotted border-brand">{name}</Text>
            </Tooltip>
          );
        },
      }),
      accessor('source_link', {
        header: 'Source',
        enableSorting: false,
        size: 140,
        cell: ({ row }) => {
          const { source_link } = row.original;
          if (!source_link) return <Text>-</Text>;
          return <ChangesetBadge href={source_link} />;
        },
      }),
      accessor((original) => original.agent_names?.join(', '), {
        id: 'agent_names',
        header: 'Agent Names',
        enableSorting: false,
        meta: {
          filter: { type: 'text', label: 'Agent Name', placeholder: 'Filter by Agent Name' },
        },
        cell: ({ getValue }) => <Text>{getValue<string>() || '-'}</Text>,
      }),
      accessor((original) => original.agent_versions?.join(', '), {
        id: 'agent_versions',
        header: 'Agent Versions',
        enableSorting: false,
        meta: {
          filter: { type: 'text', label: 'Agent Version', placeholder: 'Filter by Agent Version' },
        },
        cell: ({ getValue }) => <Text>{getValue<string>() || '-'}</Text>,
      }),
      accessor('dataset_name', {
        header: 'Dataset Name',
        enableSorting: false,
        meta: {
          filter: { type: 'text', label: 'Dataset Name', placeholder: 'Filter by Dataset Name' },
        },
        cell: ({ row }) => <Text>{row.original.dataset_name || '-'}</Text>,
      }),
      accessor('dataset_version', {
        header: 'Dataset Version',
        enableSorting: false,
        meta: {
          filter: {
            type: 'text',
            label: 'Dataset Version',
            placeholder: 'Filter by Dataset Version',
          },
        },
        cell: ({ row }) => <Text>{row.original.dataset_version || '-'}</Text>,
      }),
      accessor((original) => original.model_names?.join(', '), {
        id: 'model_names',
        header: 'Models',
        enableSorting: false,
        meta: {
          filter: { type: 'text', label: 'Model', placeholder: 'Filter by Model' },
        },
        cell: ({ getValue }) => <Text>{getValue<string>() || '-'}</Text>,
      }),
      ...metadataKeys.map((key) =>
        accessor(
          (original) => {
            const meta = original.metadata ?? {};
            // Match the first key that lowercases to this column's key.
            const match = Object.keys(meta).find((k) => k.toLowerCase() === key);
            return match ? meta[match] : undefined;
          },
          {
            id: `metadata-${key}`,
            header: snakeCaseToTitleCase(key),
            enableSorting: false,
            cell: ({ getValue }) => {
              const raw = getValue<unknown>();
              if (raw == null) return <Text>-</Text>;
              const str = typeof raw === 'object' ? JSON.stringify(raw) : String(raw);
              if (str.length <= 50) return <Text>{str}</Text>;
              return (
                <Tooltip
                  slotContent={<Text kind="body/regular/sm">{str}</Text>}
                  className={tooltipClassName}
                  side="bottom"
                >
                  <Text className="cursor-default">{str.slice(0, 50)}…</Text>
                </Tooltip>
              );
            },
          }
        )
      ),
      ...evaluatorNames.map((name) => {
        const title = snakeCaseToTitleCase(name);
        return accessor((original) => original.aggregate_scores?.[name]?.mean, {
          id: `evaluator-${name}`,
          header: `Avg ${title}`,
          enableSorting: true,
          meta: { title: false, filter: numberRangeFilter(`Avg ${title}`) },
          size: 140,
          cell: ({ row }) => {
            const score = row.original.aggregate_scores?.[name];
            return (
              <MeanValueTooltipCell
                label={title}
                count={score?.count}
                runCount={row.original.run_count}
                countsMissingAsZero
              >
                {formatEvaluatorScore(score?.mean)}
              </MeanValueTooltipCell>
            );
          },
        });
      }),
      accessor((original) => original.cost_usd?.mean, {
        id: 'cost_usd',
        header: 'Avg Cost',
        enableSorting: true,
        meta: { title: false, filter: numberRangeFilter('Avg Cost') },
        cell: ({ row }) => {
          const { cost_usd, run_count } = row.original;
          return (
            <MeanValueTooltipCell label="cost" count={cost_usd?.count} runCount={run_count}>
              {cost_usd?.mean != null ? `$${cost_usd.mean.toFixed(3)}` : '-'}
            </MeanValueTooltipCell>
          );
        },
      }),
      accessor((original) => original.latency_ms?.mean, {
        id: 'latency_ms',
        header: 'Avg Latency',
        meta: { title: false, filter: numberRangeFilter('Avg Latency') },
        enableSorting: true,
        cell: ({ row }) => {
          const { latency_ms, run_count } = row.original;
          return (
            <MeanValueTooltipCell label="latency" count={latency_ms?.count} runCount={run_count}>
              {latency_ms?.mean != null ? formatDurationMs(latency_ms.mean) : '-'}
            </MeanValueTooltipCell>
          );
        },
      }),
      accessor((original) => original.latency_ms?.sum, {
        id: 'total_latency_ms',
        header: 'Total latency',
        enableSorting: true,
        meta: { title: false, filter: numberRangeFilter('Total latency') },
        cell: ({ row }) => <Text>{formatDurationMs(row.original.latency_ms?.sum)}</Text>,
      }),
      accessor((original) => original.tokens?.mean, {
        id: 'tokens',
        header: 'Avg Tokens',
        enableSorting: true,
        meta: { title: false, filter: numberRangeFilter('Avg Tokens') },
        cell: ({ row }) => {
          const { tokens, run_count: runCount } = row.original;
          return (
            <MeanValueTooltipCell
              label="tokens (input + output)"
              count={tokens?.count}
              runCount={runCount}
            >
              {tokens?.mean != null ? Math.round(tokens.mean).toLocaleString() : '-'}
            </MeanValueTooltipCell>
          );
        },
      }),
      accessor((original) => original.tokens?.sum, {
        id: 'total_tokens',
        // Sum of per-test-case means: the rollup is deliberately k-invariant, so this is one pass
        // through the dataset, not total tokens burned across repeated attempts.
        header: 'Total tokens (per test case)',
        enableSorting: true,
        meta: { title: false, filter: numberRangeFilter('Total tokens (per test case)') },
        cell: ({ row }) => {
          const { tokens } = row.original;
          return <Text>{tokens?.sum != null ? Math.round(tokens.sum).toLocaleString() : '-'}</Text>;
        },
      }),
      accessor((original) => original.test_case_count, {
        id: 'test_case_count',
        header: 'Total test cases',
        enableSorting: true,
        cell: ({ row }) => <Text>{String(row.original.test_case_count ?? 0)}</Text>,
      }),
      // Written as metadata at publish time, so it is a string and only present for runs that
      // published successfully. Not sortable: the API sorts metadata lexically, which would order
      // "9" after "10", and the list pages at 100 so a client-side sort would lie across pages.
      accessor((original) => evalDurationMs(original.metadata), {
        id: 'eval_duration',
        header: 'Duration',
        enableSorting: false,
        cell: ({ getValue }) => <Text>{formatDurationMs(getValue<number | undefined>())}</Text>,
      }),
      accessor('created_at', {
        header: 'Created',
        size: 200,
        enableSorting: true,
        meta: { filter: dateTimeFilter('Created At') },
        cell: ({ row }) =>
          row.original.created_at ? (
            <RelativeTime datetime={row.original.created_at} />
          ) : (
            <Text>-</Text>
          ),
      }),
      // Filter-only columns (hidden via columnVisibility above).
      accessor(() => '', {
        id: 'created_by',
        header: 'Created By',
        enableSorting: false,
        enableHiding: false,
        meta: {
          filter: { type: 'text', label: 'Created By', placeholder: 'Filter by Created By' },
        },
      }),
      rowActionsColumn({
        size: ROW_ACTIONS_COLUMN_SIZE,
        enableResizing: false,
        cell: ({ row }) => (
          <QuickActionsMenuRoot
            actions={[
              // Needs a reusable eval config and the agent that produced it: the form's picker is
              // agent-scoped, so an untagged row opens on a step with nothing to choose.
              ...(evaluationFilesetName(row.original) && row.original.agent_names?.[0]
                ? [
                    {
                      label: 'New evaluation from this configuration',
                      icon: <Repeat2 className="fill-none" size={16} />,
                      onSelect: () => setRerunSource(row.original),
                    },
                  ]
                : []),
              {
                // `fill-none` overrides the menu's cloned `fill: 'solid'`, which would otherwise paint
                // the folder body over the "+"/"-" (they're drawn before the folder shape).
                label: 'Add to experiment',
                icon: <FolderPlus className="fill-none" size={16} />,
                onSelect: () =>
                  setAddToGroupState({ evaluations: [row.original], clearSelection: () => {} }),
              },
              {
                label: 'Remove from experiment',
                icon: <FolderMinus className="fill-none" size={16} />,
                onSelect: () => removeFromGroup(row.original),
              },
            ]}
          />
        ),
      }),
    ],
    [evaluatorNames, togglePin, removeFromGroup, metadataKeys]
  );

  // A recoverable sort error is handled by useSortErrorRecovery (toast + revert), and the table keeps
  // showing the last good page — so don't replace it with the full-page error for that case.
  if (error && !isRecoverableSortError) {
    return <ErrorMessage message="Failed to load experiments." />;
  }

  // When the whole unfiltered group fits on the first page, the loaded rows are the complete set, so we
  // pass them to the Pareto chart (decided upfront via `evaluation_count`) instead of a second fetch.
  const hasActiveFilter = Object.keys(dataViewState.apiFilter.filter ?? {}).length > 0;
  const groupFitsOnePage =
    group.evaluation_count != null &&
    group.evaluation_count <= pageSize &&
    page === 1 &&
    !dataViewState.debouncedSearchBar &&
    !hasActiveFilter;
  const completeEvaluationSet = groupFitsOnePage && !isLoading ? orderedData : undefined;
  const preloadPending = groupFitsOnePage && isLoading;

  return (
    <>
      {trendVisible && (
        <div className="mb-4">
          {/* Key by group id for the same reason as the Pareto chart: re-seed the selected metric
              when navigating between groups without a route remount. */}
          <ExperimentTrendChart
            key={group.id}
            workspace={workspace}
            group={group}
            preloadedEvaluations={completeEvaluationSet}
            preloadPending={preloadPending}
          />
        </div>
      )}
      {paretoVisible && (
        <div className="mb-4">
          {/* Key by group id so the axis selection resets (re-seeds from the new group's saved
              config) when navigating between groups without a route remount. */}
          <ExperimentParetoChart
            key={group.id}
            workspace={workspace}
            group={group}
            preloadedEvaluations={completeEvaluationSet}
            preloadPending={preloadPending}
          />
        </div>
      )}
      <StudioDataView
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        searchField="name"
        onRowClick={(row, _index, event) =>
          openRow(event, getEvaluationDetailRoute(workspace, experimentName, row.name))
        }
        renderBulkActions={({ selectedRows, table }) => (
          <Button
            kind="tertiary"
            color="neutral"
            size="small"
            onClick={() =>
              setAddToGroupState({
                evaluations: selectedRows,
                clearSelection: () => table.resetRowSelection(),
              })
            }
          >
            <FolderPlus />
            Add to experiment
          </Button>
        )}
        toolbarSlotStart={
          columnLayout.hasUnsavedLayout ? (
            <Button
              className="shrink-0"
              kind="primary"
              disabled={columnLayout.isSaving}
              onClick={columnLayout.save}
            >
              {columnLayout.isSaving ? 'Saving…' : 'Save columns'}
            </Button>
          ) : undefined
        }
        toolbarSlotEnd={
          <EditColumnsMenu
            kind="secondary"
            showChevron={false}
            // EditColumnsMenu exposes no width control for its dropdown, so this zero-height
            // spacer sets a min width on the menu (which sizes to its widest child).
            slotContent={<div aria-hidden className="h-0 w-[230px]" />}
          >
            <>
              <Columns3 />
              <span className="hide-mobile">Columns</span>
            </>
          </EditColumnsMenu>
        }
        attributes={{
          DataViewRoot: {
            data: orderedData,
            totalCount,
            requestStatus: isLoading ? 'loading' : undefined,
            className: 'experiment-data-view',
          },
          DataViewTableContent: {
            enableColumnReordering: true,
            renderEmptyState: ({ hasFiltersApplied, hasSearchApplied }) =>
              hasFiltersApplied || hasSearchApplied ? (
                <EntityEmptyState
                  entity="evaluationResults"
                  variant="no-results"
                  onClearFilters={dataViewState.resetFilters}
                />
              ) : (
                <EntityEmptyState entity="evaluationResults" variant="first-use" />
              ),
          },
        }}
      />
      {addToGroupState && (
        <AddToGroupModal
          open
          onClose={() => setAddToGroupState(null)}
          onSuccess={() => addToGroupState.clearSelection()}
          workspace={workspace}
          evaluations={addToGroupState.evaluations}
        />
      )}
      {rerunSource && (
        <SubmitEvaluationModal
          open
          onClose={() => setRerunSource(null)}
          workspace={workspace}
          // Agents are discovered from the run's own telemetry; the form still lets the user pick
          // when the row was never tagged with one.
          agent={rerunSource.agent_names?.[0]}
          sourceEvaluation={rerunSource.name}
        />
      )}
    </>
  );
};
