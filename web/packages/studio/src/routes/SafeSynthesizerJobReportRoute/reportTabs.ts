// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CheckmarkBadge, Grid, IconDocument, LockClosed } from '@nemo/icons/icons';
import type {
  EvaluationReportData,
  MetricScore,
} from '@studio/routes/SafeSynthesizerJobReportRoute/reportDataTypes';

export const REPORT_TABS = {
  overview: { label: 'Overview', Icon: IconDocument },
  'synthetic-quality': { label: 'Synthetic Quality', Icon: CheckmarkBadge },
  'data-privacy': { label: 'Data Privacy', Icon: LockClosed },
  'training-data': { label: 'Training Data', Icon: Grid },
} as const;

export type ReportTab = keyof typeof REPORT_TABS;

export const isScoreAvailable = (score: MetricScore | null | undefined): boolean =>
  score !== null && score !== undefined && score.grade !== 'Unavailable';

/** Tabs the library report would show for this data; without data, the summary-backed tabs. */
export const visibleReportTabs = (data: EvaluationReportData | null): ReportTab[] => {
  if (data === null) {
    return ['overview', 'synthetic-quality', 'data-privacy'];
  }
  const tabs: ReportTab[] = ['overview'];
  if (isScoreAvailable(data.scores.synthetic_quality)) {
    tabs.push('synthetic-quality');
  }
  if (isScoreAvailable(data.scores.data_privacy) || data.pii_replacement !== null) {
    tabs.push('data-privacy');
  }
  if (data.columns.length > 0) {
    tabs.push('training-data');
  }
  return tabs;
};
