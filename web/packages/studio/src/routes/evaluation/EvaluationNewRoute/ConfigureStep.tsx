// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Text } from '@nvidia/foundations-react-core';
import { ConfigNameField } from '@studio/routes/evaluation/EvaluationNewRoute/ConfigNameField';
import { DatasetPanel } from '@studio/routes/evaluation/EvaluationNewRoute/DatasetPanel';
import { MetricPanel } from '@studio/routes/evaluation/EvaluationNewRoute/MetricPanel';
import { useDatasetReady } from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetReady';
import { FC } from 'react';

/**
 * The configuration itself: what gets saved to a fileset and reused later.
 *
 * Only reached when authoring. A saved configuration skips this step entirely
 * rather than showing it back read-only -- there is nothing to do here once it
 * exists, and the model it runs against is asked for on the next step.
 *
 * Metrics are absent until the dataset resolves rather than shown disabled: the
 * choices depend on which fields are bound, so before that there is nothing
 * meaningful to read there.
 *
 * Section headings are unnumbered: this is a step of a wizard that already shows
 * a numbered stepper, and a second sequence inside it competes with it.
 */
export const ConfigureStep: FC = () => {
  const datasetReady = useDatasetReady();

  return (
    <Stack gap="density-2xl" className="w-full min-w-0">
      <ConfigNameField />

      <Stack gap="density-sm" className="min-w-0">
        <Text kind="label/bold/xl">Dataset</Text>
        <DatasetPanel />
      </Stack>

      {datasetReady ? (
        <Stack gap="density-sm" className="min-w-0">
          <Text kind="label/bold/xl">Evaluation Metrics</Text>
          <MetricPanel />
        </Stack>
      ) : null}
    </Stack>
  );
};
