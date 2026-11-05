// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChartLegend } from '@nemo/common/src/components/charts/ChartLegend';
import { CircleTick } from '@nemo/icons/icons';
import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { HeadToHeadChart } from '@studio/components/charts/EvaluationCompareChart/HeadToHeadChart';
import { ImprovementChart } from '@studio/components/charts/EvaluationCompareChart/ImprovementChart';
import { ScoresChart } from '@studio/components/charts/EvaluationCompareChart/ScoresChart';
import { useComparedEvaluations } from '@studio/components/charts/EvaluationCompareChart/useComparedEvaluations';
import {
  buildImprovementRows,
  buildMetricValueRows,
  describeComparison,
  isEvaluatorMetric,
  toComparedEvaluations,
} from '@studio/components/charts/EvaluationCompareChart/utils';
import { deriveTrendMetrics } from '@studio/components/charts/ExperimentTrendChart/utils';
import { MetricSelect } from '@studio/components/charts/MetricSelect';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';
import { type FC, type ReactNode, useMemo, useState } from 'react';

interface EvaluationCompareChartProps {
  workspace: string;
  evaluations: readonly EvaluationRow[];
}

const Section: FC<{ title: string; description: string; children: ReactNode }> = ({
  title,
  description,
  children,
}) => (
  <Stack gap="density-md">
    <Stack gap="density-xxs">
      <Text kind="label/bold/md">{title}</Text>
      <Text kind="body/regular/xs" color="subtle">
        {description}
      </Text>
    </Stack>
    {children}
  </Stack>
);

const Placeholder: FC<{ children: ReactNode }> = ({ children }) => (
  <Text kind="body/regular/sm" color="subtle">
    {children}
  </Text>
);

export const EvaluationCompareChart: FC<EvaluationCompareChartProps> = ({
  workspace,
  evaluations,
}) => {
  const compared = useMemo(() => toComparedEvaluations(evaluations), [evaluations]);
  const [baselineId, setBaselineId] = useState<string | undefined>(undefined);
  const baseline = compared.find(({ row }) => row.id === baselineId) ?? compared[0];
  const candidates = useMemo(
    () => compared.filter((evaluation) => evaluation !== baseline),
    [compared, baseline]
  );

  const metrics = useMemo(() => deriveTrendMetrics(compared.map(({ row }) => row)), [compared]);
  const valueRows = useMemo(() => buildMetricValueRows(compared, metrics), [compared, metrics]);
  const scoreRows = valueRows.filter(({ metric }) => isEvaluatorMetric(metric));
  const improvementRows = useMemo(
    () =>
      baseline
        ? buildImprovementRows(
            valueRows,
            baseline.row.id,
            candidates.map(({ row }) => row.id)
          )
        : [],
    [valueRows, baseline, candidates]
  );

  const evaluatorMetrics = scoreRows.map(({ metric }) => metric);
  const [headToHeadMetricId, setHeadToHeadMetricId] = useState<string | undefined>(undefined);
  const headToHeadMetric =
    evaluatorMetrics.find(({ id }) => id === headToHeadMetricId) ?? evaluatorMetrics[0];

  // The pairing runs server-side: one compare request per candidate against the selected
  // baseline and evaluator, so this chart, the CLI, and the API agree by construction.
  const candidateNames = useMemo(() => candidates.map(({ row }) => row.name), [candidates]);
  const comparisons = useComparedEvaluations(
    workspace,
    baseline?.row.name,
    candidateNames,
    headToHeadMetric?.id
  );
  const headToHeadRows = useMemo(() => {
    if (!baseline || !headToHeadMetric || comparisons.isLoading) return [];
    return comparisons.comparisons.map(({ name, result }) => ({
      name,
      better: result?.better ?? 0,
      same: result?.same ?? 0,
      worse: result?.worse ?? 0,
    }));
  }, [baseline, headToHeadMetric, comparisons]);
  const hasPairedTestCases = headToHeadRows.some((row) => row.better + row.same + row.worse > 0);

  if (!baseline) return null;

  const legendItems = compared.map(({ row, color }) => ({
    id: row.id,
    label: row === baseline.row ? `${row.name} (baseline)` : row.name,
    color,
  }));

  const renderHeadToHead = () => {
    if (!headToHeadMetric) return <Placeholder>No evaluator scores to compare.</Placeholder>;
    if (comparisons.isError) return <Placeholder>Could not load the paired comparison.</Placeholder>;
    if (comparisons.isLoading) {
      return (
        <Flex aria-busy align="center" gap="density-md">
          <CircleTick width={16} height={16} className="animate-spin text-brand" />
          <Placeholder>Comparing test case results…</Placeholder>
        </Flex>
      );
    }
    if (!hasPairedTestCases) {
      return <Placeholder>No test cases were scored by both runs.</Placeholder>;
    }
    return (
      <Stack gap="density-sm">
        <HeadToHeadChart rows={headToHeadRows} />
        {comparisons.comparisons.map(({ name, result }) =>
          result ? (
            <Stack key={name} gap="density-xxs">
              <Text kind="body/regular/xs" color="subtle">
                {`${name}: ${describeComparison(result)}`}
              </Text>
              {result.interval_caveat ? (
                <Text kind="body/regular/xs" color="subtle">
                  {result.interval_caveat}
                </Text>
              ) : null}
            </Stack>
          ) : null
        )}
      </Stack>
    );
  };

  return (
    <Stack gap="density-xl">
      <Flex wrap="wrap" align="center" justify="between" gap="density-lg">
        <MetricSelect
          label="Baseline"
          value={baseline.row.id}
          metrics={compared.map(({ row }) => ({ id: row.id, label: row.name }))}
          onChange={setBaselineId}
          triggerClassName="w-56"
        />
        <ChartLegend items={legendItems} interactive={false} justify="end" />
      </Flex>
      <Section title="Scores" description="Mean score per evaluator.">
        {scoreRows.length > 0 ? (
          <ScoresChart evaluations={compared} rows={scoreRows} />
        ) : (
          <Placeholder>None of these evaluations have evaluator scores.</Placeholder>
        )}
      </Section>
      <Section
        title="Change vs baseline"
        description="Relative change per metric. Right is better: higher scores, lower cost, latency, tokens and duration."
      >
        {improvementRows.length > 0 ? (
          <ImprovementChart baseline={baseline} candidates={candidates} rows={improvementRows} />
        ) : (
          <Placeholder>No metrics shared with the baseline.</Placeholder>
        )}
      </Section>
      <Section
        title="Per test case"
        description="How each run scored on the same test cases as the baseline. Repeated attempts are averaged server-side, with a 95% bootstrap interval and P(candidate better) for the selected evaluator."
      >
        {evaluatorMetrics.length > 1 && headToHeadMetric && (
          <MetricSelect
            label="Evaluator"
            value={headToHeadMetric.id}
            metrics={evaluatorMetrics}
            onChange={setHeadToHeadMetricId}
            triggerClassName="w-44"
          />
        )}
        {renderHeadToHead()}
      </Section>
    </Stack>
  );
};
