// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_SOURCE } from '@studio/components/dataViews/JobsDataView/constants';
import { iconColorClass } from '@studio/routes/constants';
import { ChartBar, LayoutList, Sliders, Sparkles } from 'lucide-react';
import type { ReactNode } from 'react';

export const SOURCE_DISPLAY: Record<string, { label: string; icon: ReactNode }> = {
  [JOB_SOURCE.CUSTOMIZATION]: {
    label: 'Fine-tuning',
    icon: <Sliders className={iconColorClass} size={14} />,
  },
  [JOB_SOURCE.DATA_DESIGNER]: {
    label: 'Data Designer',
    icon: <LayoutList className={iconColorClass} size={14} />,
  },
  [JOB_SOURCE.SAFE_SYNTHESIZER]: {
    label: 'Safe Synthesizer',
    icon: <Sparkles className={iconColorClass} size={14} />,
  },
  [JOB_SOURCE.EVALUATOR_METRICS]: {
    label: 'Evaluator',
    icon: <ChartBar className={iconColorClass} size={14} />,
  },
};
