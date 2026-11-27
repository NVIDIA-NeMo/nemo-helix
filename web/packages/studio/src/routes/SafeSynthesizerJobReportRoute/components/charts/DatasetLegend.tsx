// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChartLegend } from '@nemo/common/src/components/charts/ChartLegend';
import {
  DATASET_COLORS,
  DATASET_LABELS,
} from '@studio/routes/SafeSynthesizerJobReportRoute/chartData';
import type { FC } from 'react';

type Dataset = keyof typeof DATASET_COLORS;

const legendItems = (datasets: readonly Dataset[]) =>
  datasets.map((dataset) => ({
    id: dataset,
    label: DATASET_LABELS[dataset],
    color: DATASET_COLORS[dataset],
  }));

const TRAINING_AND_SYNTHETIC: readonly Dataset[] = ['training', 'synthetic'];

export const DatasetLegend: FC<{ datasets?: readonly Dataset[] }> = ({
  datasets = TRAINING_AND_SYNTHETIC,
}) => <ChartLegend items={legendItems(datasets)} interactive={false} justify="end" />;
