// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KeyValueGrid } from '@nemo/common/src/components/KeyValueGrid';
import {
  Anchor,
  Badge,
  Button,
  Flex,
  Modal,
  Stack,
  Table,
  Text,
} from '@nvidia/foundations-react-core';
import { DetailPanel } from '@studio/components/DetailPanel';
import {
  ARM_NAMES,
  comparisonMetrics,
  tokenUsage,
  type ArmName,
  type ComparisonArm,
  type ComparisonReport,
  type MetricObservation,
} from '@studio/routes/evaluation/SkillComparisonRoute/report';
import { getAgentEvaluationDetailRoute } from '@studio/routes/utils';
import { useState } from 'react';
import { Link } from 'react-router';

const labels: Record<ArmName, string> = { baseline: 'Baseline', 'with-skill': 'With Skill' };
const format = (value: number | null | undefined) =>
  value == null
    ? 'Unmeasured'
    : new Intl.NumberFormat('en-US', { maximumFractionDigits: 4 }).format(value);
const flag = (value: boolean | null | undefined) =>
  value == null ? 'Unmeasured' : value ? 'Yes' : 'No';
const deltaText = (value: number | null) =>
  value === null ? '—' : `${value > 0 ? '+' : ''}${format(value)}`;

function Observation({ observation }: { observation: MetricObservation }) {
  const { coverage, value, status } = observation;
  return (
    <Stack gap="1">
      <Text className="tabular-nums">{observation.label ?? format(value)}</Text>
      <Text kind="body/regular/sm" className="text-secondary">
        {status}
        {coverage ? ` · ${coverage.scored}/${coverage.total} scored` : ''}
      </Text>
    </Stack>
  );
}

export function MetricComparison({
  report,
  onInspect,
}: {
  report: ComparisonReport;
  onInspect: (metric: string) => void;
}) {
  const metrics = comparisonMetrics(report);
  const chartMetrics = metrics.filter(
    (row) =>
      !row.name.includes('.pass@') &&
      [row.baseline.value, row.treatment.value].some(
        (value) => value !== null && value >= 0 && value <= 1
      ) &&
      [row.baseline.value, row.treatment.value].every(
        (value) => value === null || (value >= 0 && value <= 1)
      )
  );
  return (
    <Stack gap="density-xl">
      {chartMetrics.length > 0 && (
        <DetailPanel title="Scores by Metric">
          <Stack gap="density-lg">
            <Text kind="body/regular/sm" className="text-secondary">
              Thick bars show with-skill scores; thin bars show the baseline. Only measured scores
              on a 0–1 scale appear here. Select a metric to inspect its evidence. Confidence
              intervals are not available in this report.
            </Text>
            <div className="grid grid-cols-[minmax(10rem,1fr)_minmax(12rem,3fr)] gap-x-density-lg gap-y-density-lg overflow-x-auto">
              <span />
              <div className="flex justify-between text-secondary text-xs tabular-nums" aria-hidden>
                {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
                  <span key={tick}>{tick}</span>
                ))}
              </div>
              {chartMetrics.map((metric) => (
                <div key={metric.name} className="contents">
                  <Button
                    kind="tertiary"
                    className="justify-start h-auto break-all text-left"
                    onClick={() => onInspect(metric.name)}
                  >
                    {metric.name}
                  </Button>
                  <Stack gap="2" className="min-w-0">
                    {(
                      [
                        ['with-skill', metric.treatment],
                        ['baseline', metric.baseline],
                      ] as const
                    ).map(([arm, observation]) => (
                      <Flex
                        key={arm}
                        gap="density-sm"
                        align="center"
                        aria-label={`${metric.name}, ${labels[arm]}: ${format(observation.value)}`}
                      >
                        <div className="flex-1 rounded bg-surface-sunken" aria-hidden>
                          <svg
                            viewBox="0 0 100 12"
                            preserveAspectRatio="none"
                            width="100%"
                            height={arm === 'with-skill' ? 12 : 6}
                          >
                            <rect
                              width={(observation.value ?? 0) * 100}
                              height="12"
                              rx="2"
                              fill={
                                arm === 'with-skill'
                                  ? 'var(--color-chart-data-categorical-1)'
                                  : 'var(--color-chart-data-categorical-2)'
                              }
                            />
                          </svg>
                        </div>
                        <Text kind="body/regular/sm" className="w-24 text-right tabular-nums">
                          {format(observation.value)}
                        </Text>
                      </Flex>
                    ))}
                  </Stack>
                </div>
              ))}
            </div>
          </Stack>
        </DetailPanel>
      )}
      <Stack gap="density-md">
        <Text kind="title/sm">Every Metric</Text>
        <Text kind="body/regular/sm" className="text-secondary">
          Difference is with skill minus baseline, shown only for complete measured results with
          matching inputs. A positive difference means a higher value; its desirability depends on
          the metric.
        </Text>
        {metrics.length === 0 ? (
          <Text>
            No replay metrics are available. Inspect Evidence for collection or scoring errors.
          </Text>
        ) : (
          <div className="overflow-x-auto">
            <Table
              aria-label="Metric comparison"
              className="min-w-[42rem]"
              layout="auto"
              hoverableRows
              columns={['Metric', 'Baseline', 'With Skill', 'Difference']}
              rows={metrics.map((metric) => ({
                id: metric.name,
                onRowSelect: () => onInspect(metric.name),
                cells: [
                  metric.name,
                  { children: <Observation observation={metric.baseline} />, align: 'right' },
                  { children: <Observation observation={metric.treatment} />, align: 'right' },
                  { children: deltaText(metric.delta), align: 'right', className: 'tabular-nums' },
                ],
              }))}
            />
          </div>
        )}
      </Stack>
    </Stack>
  );
}

export function TokenComparison({ report }: { report: ComparisonReport }) {
  const baseline = tokenUsage(report.arms.baseline);
  const treatment = tokenUsage(report.arms['with-skill']);
  const rows = [
    { name: 'Prompt (including cached input)', key: 'prompt' },
    { name: 'Cached input (subset of prompt)', key: 'cached' },
    { name: 'Completion', key: 'completion' },
    { name: 'Total (prompt + completion)', key: 'total' },
    { name: 'Uncached prompt + completion', key: 'uncached' },
  ] as const;
  return (
    <DetailPanel title="Token Usage">
      <Stack gap="density-md">
        <Text kind="body/regular/sm" className="text-secondary">
          Cached input is part of prompt tokens and is counted once. Older reports may retain only
          uncached prompt plus completion tokens.
        </Text>
        <div className="overflow-x-auto">
          <Table
            aria-label="Token usage"
            className="min-w-[36rem]"
            layout="auto"
            columns={['Tokens', 'Baseline', 'With Skill', 'Difference']}
            rows={rows.map(({ name, key }) => ({
              id: key,
              cells: [
                name,
                { children: format(baseline[key]), align: 'right' },
                { children: format(treatment[key]), align: 'right' },
                {
                  children: deltaText(
                    baseline[key] !== null && treatment[key] !== null
                      ? treatment[key] - baseline[key]
                      : null
                  ),
                  align: 'right',
                },
              ],
            }))}
          />
        </div>
      </Stack>
    </DetailPanel>
  );
}

type TrialScore = NonNullable<ComparisonArm['replay']>['scores'][number];
interface TrialRow {
  id: string;
  task: string;
  trial: string;
  arm: ArmName;
  scores: TrialScore[];
}
const heatColors = [
  'bg-[var(--color-chart-data-sequential-blue-1)]',
  'bg-[var(--color-chart-data-sequential-blue-2)]',
  'bg-[var(--color-chart-data-sequential-blue-3)]',
  'bg-[var(--color-chart-data-sequential-blue-4)]',
  'bg-[var(--color-chart-data-sequential-blue-5)]',
  'bg-[var(--color-chart-data-sequential-blue-6)]',
  'bg-[var(--color-chart-data-sequential-blue-7)]',
];

export function TrialHeatmap({ report }: { report: ComparisonReport }) {
  const [selection, setSelection] = useState<{ row: TrialRow; metric: string } | null>(null);
  const rows = new Map<string, TrialRow>();
  const metricNames = new Set<string>();
  for (const arm of ARM_NAMES) {
    for (const score of report.arms[arm].replay?.scores ?? []) {
      const id = JSON.stringify([arm, score.task_id, score.trial_id]);
      if (!rows.has(id))
        rows.set(id, { id, task: score.task_id, trial: score.trial_id, arm, scores: [] });
      rows.get(id)?.scores.push(score);
      score.outputs.forEach((output) => metricNames.add(`${score.metric_type}.${output.name}`));
    }
  }
  const metrics = [...metricNames].sort();
  return (
    <Stack gap="density-md">
      <Text kind="title/sm">Scores by Task and Trial</Text>
      <Text className="text-secondary" kind="body/regular/sm">
        Each row identifies a task, trial, and arm. Blue intensity shows reported values from 0 to
        1; other values use neutral cells. Unmeasured cells have no intensity. Select a cell for the
        original metric outputs and diagnostics.
      </Text>
      {rows.size === 0 ? (
        <Text>No per-trial scoring evidence was retained.</Text>
      ) : (
        <div className="overflow-x-auto">
          <Table
            aria-label="Scores by task and trial"
            layout="auto"
            columns={['Task / Trial', 'Arm', ...metrics]}
            rows={[...rows.values()].map((row) => ({
              id: row.id,
              cells: [
                {
                  children: (
                    <Stack gap="1">
                      <Text>{row.task}</Text>
                      <Text kind="body/regular/sm" className="text-secondary">
                        {row.trial}
                      </Text>
                    </Stack>
                  ),
                },
                labels[row.arm],
                ...metrics.map((metric) => {
                  const score = row.scores.find((entry) =>
                    entry.outputs.some((output) => `${entry.metric_type}.${output.name}` === metric)
                  );
                  const raw = score?.outputs.find(
                    (output) => `${score.metric_type}.${output.name}` === metric
                  )?.value;
                  const value =
                    score?.status === 'completed' || score?.status === 'partial'
                      ? typeof raw === 'number'
                        ? raw
                        : typeof raw === 'boolean'
                          ? Number(raw)
                          : null
                      : null;
                  const heat =
                    value !== null && value >= 0 && value <= 1
                      ? heatColors[Math.round(value * 6)]
                      : '';
                  const display =
                    metric.endsWith('.label') &&
                    typeof raw === 'string' &&
                    (score?.status === 'completed' || score?.status === 'partial')
                      ? raw
                      : value === null
                        ? '—'
                        : format(value);
                  return {
                    children: (
                      <Button
                        kind="tertiary"
                        className={`w-full min-w-24 tabular-nums ${heat}`}
                        onClick={() => setSelection({ row, metric })}
                        aria-label={`${row.task}, ${labels[row.arm]}, ${metric}: ${display === '—' ? 'Unmeasured' : display}`}
                      >
                        <span className="rounded bg-surface-base px-1 text-primary">{display}</span>
                      </Button>
                    ),
                  };
                }),
              ],
            }))}
          />
        </div>
      )}
      <Modal
        open={selection !== null}
        onOpenChange={(open) => {
          if (!open) setSelection(null);
        }}
        slotHeading="Trial Evidence"
        className="max-w-4xl w-[90vw]"
      >
        {selection && (
          <Stack gap="density-lg" className="max-h-[65vh] overflow-auto">
            <KeyValueGrid
              items={[
                { key: 'arm', label: 'Arm', value: labels[selection.row.arm] },
                { key: 'task', label: 'Task', value: selection.row.task },
                { key: 'trial', label: 'Trial', value: selection.row.trial },
                { key: 'metric', label: 'Metric', value: selection.metric },
              ]}
            />
            <pre className="text-sm whitespace-pre-wrap break-all">
              {JSON.stringify(
                selection.row.scores.filter((score) =>
                  selection.metric.startsWith(`${score.metric_type}.`)
                ),
                null,
                2
              )}
            </pre>
          </Stack>
        )}
      </Modal>
    </Stack>
  );
}

export function ArmEvidence({
  arm,
  name,
  workspace,
}: {
  arm: ComparisonArm;
  name: ArmName;
  workspace: string;
}) {
  return (
    <DetailPanel title={labels[name]}>
      <Stack gap="density-lg">
        <KeyValueGrid
          items={[
            { key: 'run', label: 'Live Run', value: arm.run_id },
            { key: 'job', label: 'Task Job', value: arm.job_name },
            { key: 'reward', label: 'Live Harbor Reward', value: format(arm.reward) },
            { key: 'discovery', label: 'Skill Discovered', value: flag(arm.discovery) },
            { key: 'activation', label: 'Skill Activated', value: flag(arm.activation) },
            { key: 'agent', label: 'Agent Image', value: arm.agent_image_ref, wrapValue: true },
            {
              key: 'verifier',
              label: 'Verifier Image',
              value: arm.verifier_image_ref,
              wrapValue: true,
            },
            { key: 'bundle', label: 'Local Trial Bundle', value: arm.bundle, wrapValue: true },
            {
              key: 'replay-bundle',
              label: 'Local Scoring Bundle',
              value: arm.replay?.bundle,
              wrapValue: true,
            },
          ]}
        />
        {arm.scoring_job && (
          <Anchor asChild>
            <Link to={getAgentEvaluationDetailRoute(workspace, arm.scoring_job.name)}>
              Open Scoring Job
            </Link>
          </Anchor>
        )}
        {arm.replay && (
          <Badge color={arm.replay.summary.error_count ? 'yellow' : 'gray'}>
            {arm.replay.summary.trial_count} replay trials · {arm.replay.summary.error_count} trial
            errors
          </Badge>
        )}
        {(arm.evidence_error || arm.replay_error) && (
          <Text role="status" className="text-error">
            {arm.evidence_error ?? arm.replay_error}
          </Text>
        )}
      </Stack>
    </DetailPanel>
  );
}

export function MetricEvidence({ report, metric }: { report: ComparisonReport; metric: string }) {
  return (
    <Stack gap="density-lg">
      <Text kind="body/regular/sm" className="text-secondary">
        These are the stored scorer records. Semantic views are aggregates; their supporting metric
        records remain in each arm’s scoring bundle.
      </Text>
      {ARM_NAMES.map((name) => (
        <DetailPanel title={labels[name]} key={name}>
          <pre className="text-sm whitespace-pre-wrap break-all">
            {JSON.stringify(
              {
                aggregate:
                  report.arms[name].replay?.summary.scores.scores.find(
                    (score) => score.name === metric
                  ) ?? null,
                scores:
                  report.arms[name].replay?.scores.filter(
                    (score) =>
                      !metric.startsWith('view.') && metric.startsWith(`${score.metric_type}.`)
                  ) ?? [],
              },
              null,
              2
            )}
          </pre>
        </DetailPanel>
      ))}
    </Stack>
  );
}
