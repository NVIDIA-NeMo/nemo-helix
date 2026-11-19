// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  TableExpandableCell,
  type TableExpandableCellState,
} from '@nemo/common/src/components/DataView/TableExpandableCell';
import { StatTile } from '@nemo/common/src/components/StatTile';
import { triggerDownload } from '@nemo/common/src/utils/file';
import { ChartBar, Download, Lightbulb, ListCheckmark, Warning } from '@nemo/icons/icons';
import {
  Badge,
  Banner,
  Button,
  Flex,
  Grid,
  Modal,
  Panel,
  Spinner,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import {
  StaticDataView,
  type MakeColumns,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/toolCalling/StaticDataView';
import {
  deltaDirection,
  fetchArtifactBlob,
  fetchArtifactText,
  formatDelta,
  formatMetric,
  HEADLINE_METRIC,
  CALL_SIGNAL_LABEL,
  METRIC_ROWS,
  SIGNAL_ROWS,
  type SignalRow,
  rejectionReasonRows,
  type DeltaDirection,
  type MetricRow,
  type RejectedCallExample,
  type ResultLocation,
  type ToolCallingFinding,
  type ToolCallingResults,
  type ToolCallingSuggestion,
  type ToolCallingVariant,
  type VariantId,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/toolCalling/toolCallingResults';
import { UnifiedDiff } from '@studio/routes/agents/AgentOptimizationDetailRoute/toolCalling/UnifiedDiff';
import { useMutation, useQuery } from '@tanstack/react-query';
import { type ComponentProps, type FC, memo, useCallback, useMemo, useState } from 'react';

type BadgeColor = ComponentProps<typeof Badge>['color'];
type OnExpand = (state: TableExpandableCellState) => void;

const DELTA_COLOR: Record<DeltaDirection, BadgeColor> = {
  better: 'green',
  worse: 'red',
  same: 'gray',
};

const SEVERITY_COLOR: Record<ToolCallingFinding['severity'], BadgeColor> = {
  high: 'red',
  medium: 'yellow',
  low: 'gray',
};

const SOURCE_LABEL: Record<ToolCallingFinding['source'], string> = {
  'schema-lint': 'Schema lint',
  'failure-analysis': 'Failure analysis',
};

const SUGGESTION_KIND: Record<
  ToolCallingSuggestion['kind'],
  { label: string; action: string; color: BadgeColor }
> = {
  'agent-config': { label: 'Agent config', action: 'Applies automatically', color: 'green' },
  'tool-schema': { label: 'Tool schema', action: "Send to the tool's owner", color: 'blue' },
};

const EMPTY_CELL = (
  <Text kind="body/regular/sm" className="text-secondary">
    —
  </Text>
);

const metricRowId = (row: MetricRow) => row.key;
const reasonRowId = (row: { reason: string }) => row.reason;
const exampleRowId = (row: RejectedCallExample & { index: number }) => String(row.index);
const signalRowId = (row: SignalRow) => row.key;
const findingRowId = (row: ToolCallingFinding) => row.id;

const headlineChange = (baseline: ToolCallingVariant, variant: ToolCallingVariant) => {
  const { key, format, lowerIsBetter, label } = HEADLINE_METRIC;
  const before = baseline.metrics[key];
  const after = variant.metrics[key];
  const delta = after - before;
  return {
    label,
    text: `${formatMetric(before, format)} → ${formatMetric(after, format)}`,
    delta: formatDelta(delta, format),
    direction: deltaDirection(delta, lowerIsBetter),
  };
};

const DeltaBadge: FC<{ row: MetricRow; baseline: number; value: number }> = ({
  row,
  baseline,
  value,
}) => {
  const delta = value - baseline;
  return (
    <Badge kind="outline" color={DELTA_COLOR[deltaDirection(delta, row.lowerIsBetter)]}>
      {formatDelta(delta, row.format)}
    </Badge>
  );
};

const MetricsTable: FC<{ variants: readonly ToolCallingVariant[] }> = memo(({ variants }) => {
  const baseline = variants.find((variant) => variant.id === 'baseline');
  const makeColumns = useCallback<MakeColumns<MetricRow>>(
    (col) => [
      col.display({
        id: 'metric',
        header: 'Metric',
        size: 180,
        cell: ({ row }) => <Text kind="body/semibold/sm">{row.original.label}</Text>,
      }),
      ...variants.map((variant) =>
        col.display({
          id: variant.id,
          header: variant.label,
          size: 180,
          cell: ({ row }) => {
            const value = variant.metrics[row.original.key];
            return (
              <Flex align="center" gap="density-sm" wrap="wrap">
                <Text kind="body/regular/sm">{formatMetric(value, row.original.format)}</Text>
                {baseline && variant !== baseline ? (
                  <DeltaBadge
                    row={row.original}
                    baseline={baseline.metrics[row.original.key]}
                    value={value}
                  />
                ) : null}
              </Flex>
            );
          },
        })
      ),
    ],
    [variants, baseline]
  );

  return <StaticDataView rows={METRIC_ROWS} makeColumns={makeColumns} getRowId={metricRowId} />;
});
MetricsTable.displayName = 'MetricsTable';

const RejectionReasonsTable: FC<{ variants: readonly ToolCallingVariant[] }> = memo(
  ({ variants }) => {
    const rows = useMemo(() => rejectionReasonRows(variants), [variants]);
    const makeColumns = useCallback<MakeColumns<(typeof rows)[number]>>(
      (col) => [
        col.display({
          id: 'reason',
          header: 'Reason',
          size: 280,
          cell: ({ row }) => <Text kind="body/regular/sm">{row.original.reason}</Text>,
        }),
        ...variants.map((variant) =>
          col.display({
            id: variant.id,
            header: variant.label,
            size: 140,
            cell: ({ row }) => {
              const count = row.original.counts[variant.id];
              if (!count) {
                return EMPTY_CELL;
              }
              return <Text kind="body/regular/sm">{count}</Text>;
            },
          })
        ),
      ],
      [variants]
    );

    if (!rows.length) {
      return (
        <Text kind="body/regular/sm" className="text-secondary">
          No calls were rejected.
        </Text>
      );
    }
    return <StaticDataView rows={rows} makeColumns={makeColumns} getRowId={reasonRowId} />;
  }
);
RejectionReasonsTable.displayName = 'RejectionReasonsTable';

const SignalsTable: FC<{ variants: readonly ToolCallingVariant[] }> = memo(({ variants }) => {
  const makeColumns = useCallback<MakeColumns<SignalRow>>(
    (col) => [
      col.display({
        id: 'signal',
        header: 'Signal',
        size: 180,
        cell: ({ row }) => <Text kind="body/semibold/sm">{row.original.label}</Text>,
      }),
      ...variants.map((variant) =>
        col.display({
          id: variant.id,
          header: variant.label,
          size: 180,
          cell: ({ row }) => {
            const count = variant.metrics.signals?.[row.original.key];
            if (count === undefined) {
              return EMPTY_CELL;
            }
            return <Text kind="body/regular/sm">{count}</Text>;
          },
        })
      ),
    ],
    [variants]
  );

  return <StaticDataView rows={SIGNAL_ROWS} makeColumns={makeColumns} getRowId={signalRowId} />;
});
SignalsTable.displayName = 'SignalsTable';

const ExamplesTable: FC<{ examples: readonly RejectedCallExample[]; onExpand: OnExpand }> = memo(
  ({ examples, onExpand }) => {
    const rows = useMemo(
      () => examples.map((example, index) => ({ ...example, index })),
      [examples]
    );
    const makeColumns = useCallback<MakeColumns<(typeof rows)[number]>>(
      (col) => [
        col.display({
          id: 'prompt',
          header: 'Prompt',
          size: 160,
          cell: ({ row }) => <Text kind="body/regular/sm">{row.original.prompt_id}</Text>,
        }),
        col.display({
          id: 'signals',
          header: 'Signals',
          size: 180,
          cell: ({ row }) =>
            row.original.signals.length ? (
              <Flex gap="density-xs" wrap="wrap">
                {row.original.signals.map((signal) => (
                  <Badge
                    key={signal}
                    kind="outline"
                    color={signal === 'error_text' ? 'gray' : 'red'}
                  >
                    {CALL_SIGNAL_LABEL[signal]}
                  </Badge>
                ))}
              </Flex>
            ) : (
              EMPTY_CELL
            ),
        }),
        col.display({
          id: 'rejection',
          header: 'Rejection',
          size: 320,
          cell: ({ row }) => (
            <TableExpandableCell
              content={row.original.rejection}
              title={`${row.original.prompt_id} — Rejection`}
              onExpand={onExpand}
            />
          ),
        }),
        col.display({
          id: 'arguments',
          header: 'Arguments',
          size: 320,
          cell: ({ row }) => (
            <TableExpandableCell
              content={row.original.arguments_excerpt}
              title={`${row.original.prompt_id} — Arguments (${row.original.tool})`}
              onExpand={onExpand}
            />
          ),
        }),
        col.display({
          id: 'schema-errors',
          header: 'Schema errors',
          size: 320,
          cell: ({ row }) =>
            row.original.schema_errors.length ? (
              <TableExpandableCell
                content={row.original.schema_errors.join('\n')}
                title={`${row.original.prompt_id} — Schema errors`}
                onExpand={onExpand}
              />
            ) : (
              EMPTY_CELL
            ),
        }),
      ],
      [onExpand]
    );

    return <StaticDataView rows={rows} makeColumns={makeColumns} getRowId={exampleRowId} />;
  }
);
ExamplesTable.displayName = 'ExamplesTable';

const FindingsTable: FC<{ findings: readonly ToolCallingFinding[]; onExpand: OnExpand }> = memo(
  ({ findings, onExpand }) => {
    const makeColumns = useCallback<MakeColumns<ToolCallingFinding>>(
      (col) => [
        col.display({
          id: 'severity',
          header: 'Severity',
          size: 100,
          cell: ({ row }) => (
            <Badge kind="outline" color={SEVERITY_COLOR[row.original.severity]}>
              {row.original.severity}
            </Badge>
          ),
        }),
        col.display({
          id: 'finding',
          header: 'Finding',
          size: 240,
          cell: ({ row }) => (
            <Stack gap="density-xs">
              <Text kind="body/semibold/sm">{row.original.title}</Text>
              <Text kind="body/regular/sm" className="text-secondary">
                {SOURCE_LABEL[row.original.source]}
                {row.original.tool ? ` · ${row.original.tool}` : ''}
              </Text>
            </Stack>
          ),
        }),
        col.display({
          id: 'detail',
          header: 'Detail',
          size: 300,
          cell: ({ row }) =>
            row.original.detail ? (
              <TableExpandableCell
                content={row.original.detail}
                title={`${row.original.title} — Detail`}
                onExpand={onExpand}
              />
            ) : (
              EMPTY_CELL
            ),
        }),
        col.display({
          id: 'evidence',
          header: 'Evidence',
          size: 300,
          cell: ({ row }) =>
            row.original.evidence.length ? (
              <TableExpandableCell
                content={row.original.evidence.join('\n')}
                title={`${row.original.title} — Evidence`}
                onExpand={onExpand}
              />
            ) : (
              EMPTY_CELL
            ),
        }),
      ],
      [onExpand]
    );

    if (!findings.length) {
      return (
        <Text kind="body/regular/sm" className="text-secondary">
          The analysis found nothing to report.
        </Text>
      );
    }
    return <StaticDataView rows={findings} makeColumns={makeColumns} getRowId={findingRowId} />;
  }
);
FindingsTable.displayName = 'FindingsTable';

const basename = (path: string) => path.split('/').pop() ?? path;

const SuggestionDiff: FC<{ location: ResultLocation; artifact: string }> = ({
  location,
  artifact,
}) => {
  const { data, isLoading, error } = useQuery({
    queryKey: [
      'tool-calling-artifact',
      location.workspace,
      location.fileset,
      location.indexPath,
      artifact,
    ] as const,
    queryFn: ({ signal }) => fetchArtifactText(location, artifact, signal),
  });

  if (isLoading) {
    return <Spinner size="small" aria-label={`Loading ${artifact}...`} />;
  }
  if (error) {
    return (
      <Banner kind="inline" status="error">
        Could not load {artifact}: {error.message}
      </Banner>
    );
  }
  return <UnifiedDiff diff={data ?? ''} label={artifact} />;
};

const SuggestionCard: FC<{
  suggestion: ToolCallingSuggestion;
  variants: readonly ToolCallingVariant[];
  location: ResultLocation;
}> = ({ suggestion, variants, location }) => {
  const kind = SUGGESTION_KIND[suggestion.kind];
  const baseline = variants.find((variant) => variant.id === 'baseline');
  const verifiedBy = variants.find((variant) => variant.id === suggestion.verified_by_variant);
  const change = baseline && verifiedBy ? headlineChange(baseline, verifiedBy) : null;

  const download = useMutation({
    mutationFn: () => fetchArtifactBlob(location, suggestion.artifact),
    onSuccess: (blob) => triggerDownload(blob, basename(suggestion.artifact)),
  });

  return (
    <Panel
      slotHeading={suggestion.title}
      slotIcon={<Lightbulb />}
      elevation="high"
      density="compact"
      data-testid={`suggestion-${suggestion.id}`}
    >
      <Stack gap="density-lg">
        <Flex align="center" gap="density-sm" wrap="wrap">
          <Badge kind="outline" color={kind.color}>
            {kind.label}
          </Badge>
          <Text kind="body/regular/sm" className="text-secondary">
            {kind.action}
          </Text>
        </Flex>
        {suggestion.rationale ? <Text kind="body/regular/md">{suggestion.rationale}</Text> : null}
        {verifiedBy ? (
          <Flex align="center" gap="density-sm" wrap="wrap">
            <Text kind="body/regular/sm" className="text-secondary">
              Verified by {verifiedBy.label}
              {change ? ` · ${change.label} ${change.text}` : ''}
            </Text>
            {change ? (
              <Badge kind="outline" color={DELTA_COLOR[change.direction]}>
                {change.delta}
              </Badge>
            ) : null}
          </Flex>
        ) : (
          <Text kind="body/regular/sm" className="text-secondary">
            Not verified: the run did not include a variant with this change.
          </Text>
        )}
        {suggestion.diff_artifact ? (
          <SuggestionDiff location={location} artifact={suggestion.diff_artifact} />
        ) : null}
        <Flex align="center" gap="density-md" wrap="wrap">
          <Button
            kind="secondary"
            size="small"
            onClick={() => download.mutate()}
            disabled={download.isPending}
          >
            <Download />
            Download {basename(suggestion.artifact)}
          </Button>
          {download.error ? (
            <Text kind="body/regular/sm" className="text-feedback-danger">
              {download.error.message}
            </Text>
          ) : null}
        </Flex>
      </Stack>
    </Panel>
  );
};

const HeadlineTiles: FC<{ variants: readonly ToolCallingVariant[] }> = ({ variants }) => {
  const baseline = variants.find((variant) => variant.id === 'baseline');
  const final = variants[variants.length - 1];
  if (!baseline || final === baseline) {
    return null;
  }
  const tiles = METRIC_ROWS.slice(0, 3).map((row) => {
    const before = baseline.metrics[row.key];
    const after = final.metrics[row.key];
    const delta = after - before;
    const direction = deltaDirection(delta, row.lowerIsBetter);
    return {
      row,
      value: `${formatMetric(before, row.format)} → ${formatMetric(after, row.format)}`,
      delta: formatDelta(delta, row.format),
      status: direction === 'better' ? 'success' : direction === 'worse' ? 'error' : 'neutral',
    } as const;
  });

  return (
    <Grid cols={{ base: 1, md: 3 }} gap="density-xl" className="shrink-0">
      {tiles.map(({ row, value, delta, status }) => (
        <StatTile
          key={row.key}
          variant="metric"
          label={`${row.label} (${final.label})`}
          value={value}
          trailingLabel={delta}
          trailingLabelStatus={status}
        />
      ))}
    </Grid>
  );
};

/**
 * A finished tool-calling run: how often the agent's tool calls were rejected before and after
 * each suggested fix, why they were rejected, what the analysis found, and the fixes themselves.
 */
export const ToolCallingResultsView: FC<{ results: ToolCallingResults }> = ({ results }) => {
  const { result, location } = results;
  const [expandedCell, setExpandedCell] = useState<TableExpandableCellState | null>(null);
  const baseline = result.variants.find((variant) => variant.id === 'baseline');
  const hasSignals = result.variants.some((variant) => variant.metrics.signals);
  const variantLabel = (id: VariantId) =>
    result.variants.find((variant) => variant.id === id)?.label ?? id;

  return (
    <>
      {result.recommendation ? (
        <Banner kind="inline" status="success" slotIcon={<Lightbulb />}>
          {result.recommendation}
        </Banner>
      ) : null}
      <HeadlineTiles variants={result.variants} />
      <Panel
        slotHeading="Before and after"
        slotIcon={<ChartBar />}
        elevation="high"
        density="compact"
      >
        <Stack gap="density-xl">
          <MetricsTable variants={result.variants} />
          {hasSignals ? (
            <Stack gap="density-md">
              <Text kind="body/semibold/md">How bad calls were detected</Text>
              <SignalsTable variants={result.variants} />
            </Stack>
          ) : null}
        </Stack>
      </Panel>
      <Panel
        slotHeading="Why calls were rejected"
        slotIcon={<Warning />}
        elevation="high"
        density="compact"
      >
        <Stack gap="density-xl">
          <RejectionReasonsTable variants={result.variants} />
          {baseline?.examples.length ? (
            <Stack gap="density-md">
              <Text kind="body/semibold/md">Rejected calls ({variantLabel('baseline')})</Text>
              <ExamplesTable examples={baseline.examples} onExpand={setExpandedCell} />
            </Stack>
          ) : null}
        </Stack>
      </Panel>
      <Panel slotHeading="Findings" slotIcon={<ListCheckmark />} elevation="high" density="compact">
        <FindingsTable findings={result.findings} onExpand={setExpandedCell} />
      </Panel>
      {result.suggestions.length ? (
        <Stack gap="density-lg">
          <Text kind="title/sm">Suggestions</Text>
          {result.suggestions.map((suggestion) => (
            <SuggestionCard
              key={suggestion.id}
              suggestion={suggestion}
              variants={result.variants}
              location={location}
            />
          ))}
        </Stack>
      ) : null}
      <Modal
        open={expandedCell !== null}
        onOpenChange={(open) => {
          if (!open) {
            setExpandedCell(null);
          }
        }}
        slotHeading={expandedCell?.title ?? 'Cell content'}
        className="w-[90vw] max-w-[1000px]"
      >
        <div className="max-h-[70vh] overflow-auto">
          <Text kind="body/regular/md" className="whitespace-pre-wrap">
            {expandedCell?.content}
          </Text>
        </div>
      </Modal>
    </>
  );
};
