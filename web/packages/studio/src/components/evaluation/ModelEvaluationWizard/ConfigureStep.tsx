// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Text } from '@nvidia/foundations-react-core';
import { DatasetPanel } from '@studio/components/evaluation/EvalWizard/DatasetPanel';
import { useMessagesBinding } from '@studio/components/evaluation/EvalWizard/useMessagesBinding';
import { ConfigNameField } from '@studio/components/evaluation/ModelEvaluationWizard/ConfigNameField';
import { FC } from 'react';

/**
 * What the run is evaluated against: a name to save it under, the dataset, and
 * how its columns map to the canonical fields. Only reached when authoring.
 */
export const ConfigureStep: FC = () => {
  useMessagesBinding();

  return (
    <Stack gap="density-2xl" className="w-full min-w-0">
      <Text kind="label/bold/xl">Configuration</Text>
      <ConfigNameField />
      <DatasetPanel />
    </Stack>
  );
};
