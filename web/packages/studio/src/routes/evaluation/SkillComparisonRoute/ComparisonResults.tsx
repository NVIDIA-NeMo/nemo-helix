// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KeyValueGrid } from '@nemo/common/src/components/KeyValueGrid';
import { ChevronLeft, ChevronRight } from '@nemo/icons/icons';
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
import styles from '@studio/routes/evaluation/SkillComparisonRoute/comparison.module.css';
import { MetricDetails } from '@studio/routes/evaluation/SkillComparisonRoute/MetricDetails';
import {
  ARM_LABELS,
  isNormalizedOutput,
  metricLabel,
} from '@studio/routes/evaluation/SkillComparisonRoute/presentation';
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
import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { Link } from 'react-router';

const format = (value: number | null | undefined) =>
  value == null
    ? 'Unmeasured'
    : new Intl.NumberFormat('en-US', { maximumFractionDigits: 4 }).format(value);
const flag = (value: boolean | null | undefined) =>
  value == null ? 'Unmeasured' : value ? 'Yes' : 'No';
const deltaText = (value: number | null) =>
  value === null ? '—' : `${value > 0 ? '+' : ''}${format(value)}`;

const tableNumber = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });
const tableDifference = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 2,
  signDisplay: 'exceptZero',
});

const SCORE_TICKS = [0, 0.25, 0.5, 0.75, 1];
const scorePosition = (score: number) => `${score * 100}%`;

function ScoreBar({ arm, observation }: { arm: ArmName; observation: MetricObservation }) {
  const { value, range } = observation;
  const whisker =
    arm === 'with-skill' && range && range.min >= 0 && range.max <= 1 ? range : undefined;
  const score = format(value);
  const description = whisker
    ? `${ARM_LABELS[arm]}: ${score}, observed range ${format(whisker.min)} to ${format(whisker.max)}`
    : `${ARM_LABELS[arm]}: ${score}`;
  return (
    <svg className={styles.scoreTrack} height="18" role="img" aria-label={description}>
      <title>{description}</title>
      {value !== null && (
        <rect
          className={arm === 'with-skill' ? styles.treatmentBar : styles.baselineBar}
          x="0"
          y={arm === 'with-skill' ? 2 : 7}
          width={scorePosition(value)}
          height={arm === 'with-skill' ? 14 : 4}
          rx="2"
        />
      )}
      {whisker && (
        <g className={styles.range}>
          <line x1={scorePosition(whisker.min)} x2={scorePosition(whisker.max)} y1="9" y2="9" />
          {[whisker.min, whisker.max].map((bound) => (
            <line key={bound} x1={scorePosition(bound)} x2={scorePosition(bound)} y1="3" y2="15" />
          ))}
        </g>
      )}
      <text
        className={styles.value}
        x={scorePosition(whisker?.max ?? value ?? 0)}
        dx="9"
        y="50%"
        dominantBaseline="central"
      >
        {score}
      </text>
    </svg>
  );
}

function Observation({ observation }: { observation: MetricObservation }) {
  const { coverage, value, status } = observation;
  return (
    <Stack gap="1">
      <Text className="tabular-nums">
        {observation.label ?? (value === null ? 'Unmeasured' : tableNumber.format(value))}
      </Text>
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
      !row.name.endsWith('.complete') &&
      isNormalizedOutput(row.name) &&
      [row.baseline.value, row.treatment.value].some(
        (value) => value !== null && value >= 0 && value <= 1
      ) &&
      [row.baseline.value, row.treatment.value].every(
        (value) => value === null || (value >= 0 && value <= 1)
      )
  );
  const hasRanges = chartMetrics.some(
    ({ treatment: { range } }) => range && range.min >= 0 && range.max <= 1
  );
  return (
    <Stack gap="density-xl">
      {chartMetrics.length > 0 && (
        <section className={styles.chart} aria-label="Scores by metric">
          <Stack gap="density-lg">
            <Flex justify="between" align="center" gap="density-lg">
              <Text kind="title/sm">Scores by Metric</Text>
              <Flex gap="density-lg" align="center" className="flex-wrap">
                <Flex gap="2" align="center">
                  <svg width="24" height="12" aria-hidden>
                    <rect
                      width="24"
                      height="12"
                      rx="2"
                      fill="var(--color-chart-data-categorical-1)"
                    />
                  </svg>
                  <Text>{ARM_LABELS['with-skill']}</Text>
                </Flex>
                <Flex gap="2" align="center">
                  <svg width="24" height="4" aria-hidden>
                    <rect width="24" height="4" fill="var(--text-color-secondary)" />
                  </svg>
                  <Text>{ARM_LABELS.baseline}</Text>
                </Flex>
                {hasRanges && (
                  <Flex gap="2" align="center">
                    <svg width="24" height="12" viewBox="0 0 24 12" fill="none" aria-hidden>
                      <path d="M2 2v8m0-4h20m0-4v8" stroke="currentColor" strokeWidth="1.5" />
                    </svg>
                    <Text>Variant range (min–max)</Text>
                  </Flex>
                )}
              </Flex>
            </Flex>
            {!hasRanges && (
              <Text kind="body/regular/sm" className="text-secondary">
                Range markers appear when retained trial scores vary.
              </Text>
            )}
            <div className={styles.scoreGrid}>
              <span />
              <div className={styles.axis} aria-hidden>
                {SCORE_TICKS.map((tick) => (
                  <span key={tick}>{tick}</span>
                ))}
              </div>
              {chartMetrics.map((metric) => (
                <div key={metric.name} className="contents">
                  <Button
                    kind="tertiary"
                    className="justify-start h-auto text-left text-sm"
                    aria-label={metric.name}
                    title={metric.name}
                    onClick={() => onInspect(metric.name)}
                  >
                    {metricLabel(metric.name)}
                  </Button>
                  <div className={styles.plot}>
                    <div className={styles.gridlines} aria-hidden>
                      {SCORE_TICKS.map((tick) => (
                        <span key={tick} />
                      ))}
                    </div>
                    <ScoreBar arm="with-skill" observation={metric.treatment} />
                    <ScoreBar arm="baseline" observation={metric.baseline} />
                  </div>
                </div>
              ))}
            </div>
          </Stack>
        </section>
      )}
      <Stack gap="density-md">
        <Text kind="title/sm">Every Metric</Text>
        <Text kind="body/regular/sm" className="text-secondary">
          Difference is variant minus control, shown only for complete measured results with
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
              columns={['Metric', ARM_LABELS.baseline, ARM_LABELS['with-skill'], 'Difference']}
              rows={metrics.map((metric) => ({
                id: metric.name,
                onRowSelect: () => onInspect(metric.name),
                cells: [
                  {
                    children: (
                      <Stack gap="1">
                        <Text>{metricLabel(metric.name)}</Text>
                        <Text kind="body/regular/sm" className="text-secondary">
                          {metric.name}
                        </Text>
                      </Stack>
                    ),
                  },
                  { children: <Observation observation={metric.baseline} />, align: 'right' },
                  { children: <Observation observation={metric.treatment} />, align: 'right' },
                  {
                    children: metric.delta === null ? '—' : tableDifference.format(metric.delta),
                    align: 'right',
                    className: 'tabular-nums',
                  },
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
          Usage is measured in tokens. The efficiency score ranges from 0 to 1 and increases as
          uncached prompt plus completion tokens decrease.
        </Text>
        <div className="overflow-x-auto">
          <Table
            aria-label="Token usage"
            className="min-w-[36rem]"
            layout="auto"
            columns={['Tokens', ARM_LABELS.baseline, ARM_LABELS['with-skill'], 'Difference']}
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
  styles.heat0,
  styles.heat1,
  styles.heat2,
  styles.heat3,
  styles.heat4,
  styles.heat5,
  styles.heat6,
];

function MatrixViewport({ children }: { children: ReactNode }) {
  const viewport = useRef<HTMLDivElement>(null);
  const id = useId();
  const [position, setPosition] = useState<'fits' | 'start' | 'middle' | 'end'>('fits');

  function updatePosition() {
    const element = viewport.current;
    if (!element) return;
    const limit = element.scrollWidth - element.clientWidth;
    if (limit <= 1) setPosition('fits');
    else if (element.scrollLeft <= 1) setPosition('start');
    else if (element.scrollLeft >= limit - 1) setPosition('end');
    else setPosition('middle');
  }

  useEffect(() => {
    const element = viewport.current;
    if (!element) return;
    updatePosition();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(updatePosition);
    observer.observe(element);
    if (element.firstElementChild) observer.observe(element.firstElementChild);
    return () => observer.disconnect();
  }, []);

  function scrollColumns(direction: number) {
    const element = viewport.current;
    if (element) element.scrollBy({ left: direction * element.clientWidth * 0.65 });
  }

  return (
    <Stack gap="density-sm" className="min-w-0">
      {position !== 'fits' && (
        <Flex align="center" justify="end" gap="density-sm">
          <Text kind="body/regular/sm" className="text-secondary">
            Scroll columns
          </Text>
          <Button
            kind="secondary"
            size="small"
            aria-label="Scroll columns left"
            aria-controls={id}
            disabled={position === 'start'}
            onClick={() => scrollColumns(-1)}
          >
            <ChevronLeft size="16" />
          </Button>
          <Button
            kind="secondary"
            size="small"
            aria-label="Scroll columns right"
            aria-controls={id}
            disabled={position === 'end'}
            onClick={() => scrollColumns(1)}
          >
            <ChevronRight size="16" />
          </Button>
        </Flex>
      )}
      <div
        id={id}
        className="min-w-0 w-full overflow-x-auto"
        ref={viewport}
        role="region"
        aria-label="Task results matrix"
        tabIndex={0}
        onScroll={updatePosition}
      >
        {children}
      </div>
    </Stack>
  );
}

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
  const outcomeTypes = new Set(
    [...metricNames]
      .filter((name) => isNormalizedOutput(name) && !name.endsWith('.complete'))
      .map((name) => name.split('.', 1)[0])
  );
  const metrics = [...metricNames]
    .filter(
      (name) =>
        !name.endsWith('.label') &&
        ((isNormalizedOutput(name) && !name.endsWith('.complete')) ||
          !outcomeTypes.has(name.split('.', 1)[0]))
    )
    .sort();
  return (
    <Stack gap="density-md" className="min-w-0 w-full">
      <Text kind="title/sm">Scores by Task and Trial</Text>
      <Text className="text-secondary" kind="body/regular/sm">
        Each row is one trial. Color shows the reported 0–1 score; — means unmeasured. Select a cell
        to inspect its evidence.
      </Text>
      <Flex align="center" gap="2">
        <Text kind="body/regular/sm" className="text-secondary">
          Low
        </Text>
        {heatColors.map((color, index) => (
          <span
            key={color}
            className={`${color} w-6 h-3`}
            aria-label={`Score intensity ${index}/6`}
          />
        ))}
        <Text kind="body/regular/sm" className="text-secondary">
          High
        </Text>
      </Flex>
      {rows.size === 0 ? (
        <Text>No per-trial scoring evidence was retained.</Text>
      ) : (
        <MatrixViewport>
          <Table
            aria-label="Scores by task and trial"
            className={styles.matrix}
            layout="auto"
            columns={[
              'Task / Trial',
              ...metrics.map((name) => ({
                children: <span title={name}>{metricLabel(name)}</span>,
              })),
            ]}
            rows={[...rows.values()].map((row) => ({
              id: row.id,
              cells: [
                {
                  children: (
                    <Stack gap="1">
                      <Text>{row.task}</Text>
                      <Text kind="body/regular/sm" className="text-secondary">
                        {ARM_LABELS[row.arm]} · {row.trial}
                      </Text>
                    </Stack>
                  ),
                },
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
                    value !== null && value >= 0 && value <= 1 && isNormalizedOutput(metric)
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
                      <button
                        type="button"
                        className={`${styles.heatCell} ${heat}`}
                        onClick={() => setSelection({ row, metric })}
                        aria-label={`${row.task}, ${ARM_LABELS[row.arm]}, ${metric}: ${display === '—' ? 'Unmeasured' : display}`}
                      >
                        {display}
                      </button>
                    ),
                  };
                }),
              ],
            }))}
          />
        </MatrixViewport>
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
                { key: 'arm', label: 'Arm', value: ARM_LABELS[selection.row.arm] },
                { key: 'task', label: 'Task', value: selection.row.task },
                { key: 'trial', label: 'Trial', value: selection.row.trial },
                { key: 'metric', label: 'Metric', value: selection.metric },
              ]}
            />
            {selection.row.scores
              .filter((score) => selection.metric.startsWith(`${score.metric_type}.`))
              .map((score, index) => (
                <MetricDetails key={index} name={score.metric_type} result={score} />
              ))}
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
    <DetailPanel title={ARM_LABELS[name]}>
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
      {ARM_NAMES.map((name) => {
        const replay = report.arms[name].replay;
        const aggregate = replay?.summary.scores.scores.find((score) => score.name === metric);
        const scores = replay?.scores.filter((score) => metric.startsWith(`${score.metric_type}.`));
        return (
          <DetailPanel title={ARM_LABELS[name]} key={name}>
            <Stack gap="density-lg">
              {aggregate && (
                <KeyValueGrid
                  items={[
                    {
                      key: 'value',
                      label: aggregate.score_type === 'scalar' ? 'Value' : 'Mean',
                      value: format(
                        aggregate.score_type === 'scalar' ? aggregate.value : aggregate.mean
                      ),
                    },
                    {
                      key: 'min',
                      label: 'Minimum',
                      value: aggregate.min == null ? null : format(aggregate.min),
                    },
                    {
                      key: 'max',
                      label: 'Maximum',
                      value: aggregate.max == null ? null : format(aggregate.max),
                    },
                    { key: 'count', label: 'Samples', value: aggregate.count },
                  ]}
                />
              )}
              {scores?.map((score, index) => (
                <DetailPanel title={`${score.task_id} / ${score.trial_id}`} key={index}>
                  <MetricDetails name={score.metric_type} result={score} />
                </DetailPanel>
              ))}
              {!aggregate && !scores?.length && <Text>No scoring evidence was retained.</Text>}
            </Stack>
          </DetailPanel>
        );
      })}
    </Stack>
  );
}
