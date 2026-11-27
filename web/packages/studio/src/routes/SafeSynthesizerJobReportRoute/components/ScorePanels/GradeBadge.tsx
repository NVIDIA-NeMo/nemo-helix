// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { gradeColor } from '@studio/routes/SafeSynthesizerJobReportRoute/chartData';
import { gradeForScore } from '@studio/routes/SafeSynthesizerJobReportRoute/util';
import type { FC } from 'react';

export const GradeLabel: FC<{ grade: string }> = ({ grade }) => (
  <span
    data-testid="score-grade"
    className="inline-flex w-fit items-center rounded-sm border px-density-sm text-xs font-bold"
    /* eslint-disable-next-line no-restricted-syntax */
    style={{ borderColor: gradeColor(grade) }}
  >
    {grade}
  </span>
);

export const GradeBadge: FC<{ score?: number | null }> = ({ score }) => (
  <GradeLabel grade={gradeForScore(score)} />
);
