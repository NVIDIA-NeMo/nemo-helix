// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChartBar, TextAlignLeft } from '@nemo/icons/icons';
import { Grid, Stack, Text } from '@nvidia/foundations-react-core';
import { AutocorrelationCharts } from '@studio/routes/SafeSynthesizerJobReportRoute/components/charts/AutocorrelationCharts';
import { CorrelationHeatmaps } from '@studio/routes/SafeSynthesizerJobReportRoute/components/charts/CorrelationHeatmaps';
import { DatasetLegend } from '@studio/routes/SafeSynthesizerJobReportRoute/components/charts/DatasetLegend';
import { DistributionChart } from '@studio/routes/SafeSynthesizerJobReportRoute/components/charts/DistributionChart';
import {
  DeepStructureScatter,
  TextSemanticCharts,
} from '@studio/routes/SafeSynthesizerJobReportRoute/components/charts/ProjectionCharts';
import {
  MetricCard,
  MetricGroup,
} from '@studio/routes/SafeSynthesizerJobReportRoute/components/MetricCard';
import {
  SummaryLines,
  TimeSeriesNotes,
} from '@studio/routes/SafeSynthesizerJobReportRoute/components/TimeSeriesNotes';
import {
  METRIC_COPY,
  METRIC_GROUPS,
} from '@studio/routes/SafeSynthesizerJobReportRoute/reportCopy';
import type {
  ColumnResult,
  EvaluationReportData,
} from '@studio/routes/SafeSynthesizerJobReportRoute/reportDataTypes';
import { autocorrelationSummary } from '@studio/routes/SafeSynthesizerJobReportRoute/timeSeriesSummary';
import type { FC } from 'react';

const TEXT_STRUCTURE_CHARTS = [
  ['sentence_count', 'Sentence Count'],
  ['words_per_sentence', 'Words Per Sentence'],
  ['characters_per_word', 'Characters Per Word'],
] as const;

/** Columns charted in the library report: those with a distribution, plus numeric ones without (shown empty). */
const chartedColumns = (columns: readonly ColumnResult[]): ColumnResult[] =>
  columns.filter((column) => column.distribution !== null || column.training.type === 'numeric');

const ColumnDistributionDetails: FC<{ columns: ColumnResult[] }> = ({ columns }) => (
  <>
    <DatasetLegend />
    <Grid cols={{ xs: 1, lg: 2 }} gap="density-xl">
      {columns.map((column) => (
        <DistributionChart
          key={column.name}
          title={column.name}
          distribution={column.distribution}
        />
      ))}
    </Grid>
  </>
);

export const SyntheticQualityMetrics: FC<{ data: EvaluationReportData }> = ({ data }) => {
  const { metrics } = data;
  const correlations = metrics.column_correlation_stability?.correlations ?? null;
  const projection = metrics.deep_structure_stability?.projection ?? null;
  const distributionColumns = chartedColumns(data.columns);
  const semanticProjections = Object.entries(metrics.text_semantic_similarity?.projections ?? {});
  const structureColumns = Object.entries(metrics.text_structure_similarity?.columns ?? {});
  const autocorrelation = metrics.autocorrelation_similarity;

  return (
    <Stack gap="density-2xl">
      <MetricGroup
        title={METRIC_GROUPS.tabular.title}
        description={METRIC_GROUPS.tabular.description}
        icon={<ChartBar />}
      >
        <MetricCard
          id="correlation-stability"
          copy={METRIC_COPY.column_correlation_stability}
          score={metrics.column_correlation_stability?.score ?? null}
        >
          {correlations ? <CorrelationHeatmaps matrices={correlations} /> : null}
        </MetricCard>
        <MetricCard
          id="structure-stability"
          copy={METRIC_COPY.deep_structure_stability}
          score={metrics.deep_structure_stability?.score ?? null}
        >
          {projection ? (
            <>
              <DatasetLegend />
              <DeepStructureScatter projection={projection} />
            </>
          ) : null}
        </MetricCard>
        <MetricCard
          id="distribution-stability"
          copy={METRIC_COPY.column_distribution_stability}
          score={metrics.column_distribution_stability?.score ?? null}
        >
          {distributionColumns.length ? (
            <ColumnDistributionDetails columns={distributionColumns} />
          ) : null}
        </MetricCard>
      </MetricGroup>

      <MetricGroup
        title={METRIC_GROUPS.text.title}
        description={METRIC_GROUPS.text.description}
        icon={<TextAlignLeft />}
      >
        <MetricCard
          id="semantic-similarity"
          copy={METRIC_COPY.text_semantic_similarity}
          score={metrics.text_semantic_similarity?.score ?? null}
        >
          {semanticProjections.length ? (
            <>
              <DatasetLegend />
              {semanticProjections.map(([column, columnProjection]) => (
                <TextSemanticCharts key={column} column={column} projection={columnProjection} />
              ))}
            </>
          ) : null}
        </MetricCard>
        <MetricCard
          id="structure-similarity"
          copy={METRIC_COPY.text_structure_similarity}
          score={metrics.text_structure_similarity?.score ?? null}
        >
          {structureColumns.length ? (
            <>
              <DatasetLegend />
              {structureColumns.map(([column, distributions]) => (
                <Stack key={column} gap="density-sm">
                  <Text kind="label/bold/lg">{column}</Text>
                  <Grid cols={{ xs: 1, lg: 3 }} gap="density-lg">
                    {TEXT_STRUCTURE_CHARTS.map(([key, label]) => (
                      <DistributionChart
                        key={key}
                        title={label}
                        distribution={distributions[key]}
                      />
                    ))}
                  </Grid>
                </Stack>
              ))}
            </>
          ) : null}
        </MetricCard>
      </MetricGroup>

      {autocorrelation ? (
        <MetricGroup
          title={METRIC_GROUPS.timeSeries.title}
          description={METRIC_GROUPS.timeSeries.description}
          icon={<ChartBar />}
        >
          <MetricCard
            id="autocorrelation-similarity"
            copy={METRIC_COPY.autocorrelation_similarity}
            score={autocorrelation.score}
          >
            <TimeSeriesNotes score={autocorrelation.score} details={autocorrelation.details} />
            {autocorrelation.details.profiles.length ? (
              <>
                <SummaryLines lines={autocorrelationSummary(autocorrelation.details.profiles)} />
                <DatasetLegend />
                <AutocorrelationCharts
                  profiles={autocorrelation.details.profiles}
                  overallScore={autocorrelation.score.score}
                />
              </>
            ) : null}
          </MetricCard>
        </MetricGroup>
      ) : null}
    </Stack>
  );
};
