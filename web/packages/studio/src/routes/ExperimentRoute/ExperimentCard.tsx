// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatEvaluatorScore } from '@nemo/common/src/utils/formatters';
import { useListEvaluations } from '@nemo/sdk/generated/platform/evaluations';
import type { ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { Anchor, Card, Text } from '@nvidia/foundations-react-core';
import { MetricTrend } from '@studio/components/charts/MetricTrend';
import {
  DELTA_COMPARISON_LABEL,
  formatTrendDelta,
  toTrendSeries,
  TREND_EVALUATION_LIMIT,
} from '@studio/routes/ExperimentRoute/experimentTrend';
import { Metric } from '@studio/routes/ExperimentRoute/Metric';
import { UpdatedAt } from '@studio/routes/ExperimentRoute/UpdatedAt';
import { getExperimentDetailRoute } from '@studio/routes/utils';
import { Star } from 'lucide-react';
import { type FC, useMemo } from 'react';
import { Link, useNavigate } from 'react-router';

interface ExperimentCardProps {
  group: ExperimentResponse;
  workspace: string;
}

export const ExperimentCard: FC<ExperimentCardProps> = ({ group, workspace }) => {
  const navigate = useNavigate();
  const detailRoute = getExperimentDetailRoute(workspace, group.name);
  // Without an id there is nothing to filter evaluations by, so the trend cannot be built.
  const showTrend = Boolean(group.show_evaluations_over_time) && Boolean(group.id);

  // Only experiments flagged to graph over time pay for this; the rest render the plain card.
  const { data: evaluationsPage, isPending } = useListEvaluations(
    workspace,
    { filter: { experiment_id: group.id }, page_size: TREND_EVALUATION_LIMIT },
    { query: { enabled: showTrend } }
  );

  const series = useMemo(() => toTrendSeries(evaluationsPage?.data ?? []), [evaluationsPage]);

  // react-query reports a disabled query as pending forever, so this has to be gated on
  // `showTrend` — otherwise a card whose query never runs would sit on the skeleton for good.
  const isLoadingTrend = showTrend && isPending;

  // A flagged experiment with nothing to plot yet falls back to the plain card rather than
  // rendering an empty chart. Kept while the query is in flight so the card settles once,
  // into the skeleton, instead of flashing the count and then swapping to a chart.
  const renderTrend = showTrend && (isLoadingTrend || series.length > 0);

  return (
    // The card is not itself a button: the trend it can contain has its own controls, and a
    // button may not nest interactive content — assistive tech either hides the inner controls
    // or exposes them unreliably. The name is the real link, so keyboard and screen reader
    // users navigate through it; the card-wide click is a mouse-only convenience on top.
    <Card
      interactive
      attributes={{ CardContent: { className: 'flex flex-row items-center gap-6 p-6' } }}
      onClick={() => navigate(detailRoute)}
    >
      {/* Main info */}
      <div className="flex flex-col items-start gap-2 flex-1">
        <div className="flex items-center gap-2">
          {group.is_favorite && (
            <Star
              size={24}
              className="text-brand shrink-0"
              fill="currentColor"
              aria-label="Favorite"
            />
          )}
          <Anchor asChild>
            <Link to={detailRoute} className="no-underline hover:underline">
              <Text kind="title/sm">{group.name}</Text>
            </Link>
          </Anchor>
        </div>
        {group.description && (
          <Text kind="body/regular/sm" className="text-secondary">
            {group.description}
          </Text>
        )}
        <div className="flex items-center gap-4">
          {group.updated_at && <UpdatedAt datetime={group.updated_at} />}
        </div>
      </div>

      {/* Stats, or the trendline for experiments flagged to graph over time. */}
      {renderTrend ? (
        <div className="w-1/2 shrink-0">
          <MetricTrend
            label={group.name}
            series={series}
            isPending={isLoadingTrend}
            chartHeight={48}
            comparisonLabel={DELTA_COMPARISON_LABEL}
            formatValue={formatEvaluatorScore}
            formatDelta={formatTrendDelta}
          />
        </div>
      ) : (
        <div className="flex shrink-0 items-center gap-6">
          <Metric title="Evaluations" value={String(group.evaluation_count ?? 0)} />
        </div>
      )}
    </Card>
  );
};
