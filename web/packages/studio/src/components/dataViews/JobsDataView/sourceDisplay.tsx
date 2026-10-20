// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_SOURCE } from '@studio/components/dataViews/JobsDataView/constants';
import { iconColorClass } from '@studio/routes/constants';
import { ChartBar, Cog, LayoutList, Sliders, Sparkles } from 'lucide-react';
import type { ReactNode } from 'react';

interface SourceDisplay {
  readonly label: string;
  readonly icon: ReactNode;
}

const SOURCE_DISPLAY: Record<string, SourceDisplay> = {
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

const FALLBACK_ICON = <Cog className={iconColorClass} size={14} />;

/** Unknown sources fall back to their raw name and a generic icon. */
export const getSourceDisplay = (source: string): SourceDisplay =>
  SOURCE_DISPLAY[source] ?? { label: source, icon: FALLBACK_ICON };

/** Tooltip for a source cell: the label it shows, not the raw value the API reports. */
export const getSourceTitle = (value: unknown): string | undefined =>
  typeof value === 'string' && value ? getSourceDisplay(value).label : undefined;
