// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChartLegend } from '@nemo/common/src/components/charts/ChartLegend';
import { Button, Text } from '@nvidia/foundations-react-core';
import { HeadToHeadChart } from '@studio/components/charts/EvaluationCompareChart/HeadToHeadChart';
import { ImprovementChart } from '@studio/components/charts/EvaluationCompareChart/ImprovementChart';
import { ScoresChart } from '@studio/components/charts/EvaluationCompareChart/ScoresChart';
import { useComparedSessions } from '@studio/components/charts/EvaluationCompareChart/useComparedSessions';
import {
  buildImprovementRows,
  buildMetricValueRows,
  evaluatorNameOf,
  headToHead,
  isEvaluatorMetric,
  meanScoreByTestCase,
  toComparedEvaluations,
} from '@studio/components/charts/EvaluationCompareChart/utils';
import { deriveTrendMetrics } from '@studio/components/charts/ExperimentTrendChart/utils';
import { MetricSelect } from '@studio/components/charts/MetricSelect';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';
import { Loader2, X } from 'lucide-react';
import { type FC, type ReactNode, useMemo, useState } from 'react';

interface EvaluationCompareChartProps {
  workspace: string;
  evaluations: readonly EvaluationRow[];
  onClose: () => void;
}

const Section: FC<{ title: string; description: string; children: ReactNode }> = ({
  title,
  description,
  children,
}) => (
  <section className="flex flex-col gap-2">
    <div className="flex flex-col gap-0.5">
      <Text kind="label/bold/md">{title}</Text>
      <Text kind="body/regular/xs" color="subtle">
        {description}
      </Text>
    </div>
    {children}
  </section>
);

const Placeholder: FC<{ children: ReactNode }> = ({ children }) => (
  <Text kind="body/regular/sm" color="subtle">
    {children}
  </Text>
);

export const EvaluationCompareChart: FC<EvaluationCompareChartProps> = ({
  workspace,
  evaluations,
  onClose,
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

  const evaluationNames = useMemo(() => compared.map(({ row }) => row.name), [compared]);
  const sessions = useComparedSessions(workspace, evaluationNames);
  const headToHeadRows = useMemo(() => {
    if (!baseline || !headToHeadMetric || sessions.isLoading) return [];
    const evaluatorName = evaluatorNameOf(headToHeadMetric);
    const scoresOf = (name: string) =>
      meanScoreByTestCase(sessions.sessionsByEvaluation.get(name) ?? [], evaluatorName);
    const baselineScores = scoresOf(baseline.row.name);
    return candidates.map(({ row }) => ({
      name: row.name,
      ...headToHead(baselineScores, scoresOf(row.name)),
    }));
  }, [baseline, candidates, headToHeadMetric, sessions]);
  const hasPairedTestCases = headToHeadRows.some((row) => row.better + row.same + row.worse > 0);

  if (!baseline) return null;

  const legendItems = compared.map(({ row, color }) => ({
    id: row.id,
    label: row === baseline.row ? `${row.name} (baseline)` : row.name,
    color,
  }));

  const renderHeadToHead = () => {
    if (!headToHeadMetric) return <Placeholder>No evaluator scores to compare.</Placeholder>;
    if (sessions.isError) return <Placeholder>Could not load test case results.</Placeholder>;
    if (sessions.isLoading) {
      return (
        <div aria-busy className="flex items-center gap-2">
          <Loader2 width={16} height={16} className="animate-spin text-brand" />
          <Placeholder>Loading test case results…</Placeholder>
        </div>
      );
    }
    if (!hasPairedTestCases) {
      return <Placeholder>No test cases were scored by both runs.</Placeholder>;
    }
    return <HeadToHeadChart rows={headToHeadRows} />;
  };

  return (
    <div className="flex flex-col gap-4 rounded border border-base bg-surface p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <Text kind="title/xs">{`Comparing ${compared.length} evaluations`}</Text>
        <div className="flex flex-wrap items-center gap-4">
          <MetricSelect
            label="Baseline"
            value={baseline.row.id}
            metrics={compared.map(({ row }) => ({ id: row.id, label: row.name }))}
            onChange={setBaselineId}
            triggerClassName="w-56"
          />
          <Button
            kind="tertiary"
            color="neutral"
            size="small"
            aria-label="Close comparison"
            onClick={onClose}
          >
            <X />
          </Button>
        </div>
      </div>
      <ChartLegend items={legendItems} interactive={false} justify="end" />
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
        description="How each run scored on the same test cases as the baseline. Repeated attempts are averaged."
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
        {sessions.truncated.length > 0 && (
          <Placeholder>
            {`Only the first 1,000 sessions of ${sessions.truncated.join(', ')} are counted.`}
          </Placeholder>
        )}
        {renderHeadToHead()}
      </Section>
    </div>
  );
};
