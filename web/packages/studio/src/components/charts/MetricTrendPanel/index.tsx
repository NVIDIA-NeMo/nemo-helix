// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Button,
  PanelContent,
  PanelHeader,
  PanelRoot,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { MetricTrend, type MetricTrendProps } from '@studio/components/charts/MetricTrend';
import type { FC } from 'react';

export type {
  MetricTrendPoint,
  MetricTrendProps,
  MetricTrendSeries,
} from '@studio/components/charts/MetricTrend';

interface Props extends Omit<MetricTrendProps, 'label' | 'className'> {
  title: string;
  description?: string;
  onViewClick?: () => void;
  viewLabel?: string;
}

/**
 * A titled panel around {@link MetricTrend}. Use the trend on its own when the surrounding
 * container already names what is being measured.
 */
export const MetricTrendPanel: FC<Props> = ({
  title,
  description,
  onViewClick,
  viewLabel = 'View',
  ...trend
}) => (
  <PanelRoot elevation="mid">
    <PanelHeader className="items-start">
      <Stack gap="density-xs" className="min-w-0 flex-1">
        <Text kind="label/bold/xl">{title}</Text>
        {description && (
          <Text kind="body/regular/md" className="text-secondary">
            {description}
          </Text>
        )}
      </Stack>
      {onViewClick && (
        <Button kind="tertiary" size="small" className="shrink-0" onClick={onViewClick}>
          {viewLabel}
        </Button>
      )}
    </PanelHeader>

    <PanelContent>
      {/* The chart bleeds into the panel's right and bottom padding; the value column pads
          itself back so it stays optically centered against the trendline. */}
      <MetricTrend {...trend} label={title} className="-mb-density-2xl -mr-density-2xl" />
    </PanelContent>
  </PanelRoot>
);
