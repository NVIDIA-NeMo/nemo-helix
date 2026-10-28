// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Panel, Stack, Text } from '@nvidia/foundations-react-core';
import { JudgeModelSelect } from '@studio/components/evaluation/JudgeModelSelect';
import { LiveTestPanel } from '@studio/components/evaluation/ModelEvaluationWizard/LiveTestPanel';
import { type EvaluationFormValues } from '@studio/components/evaluation/ModelEvaluationWizard/types';
import { useMessagesBinding } from '@studio/components/evaluation/ModelEvaluationWizard/useMessagesBinding';
import { FC } from 'react';

/**
 * The run itself: which model to evaluate, and an optional single-row check.
 *
 * Kept apart from the configuration because it is the only thing chosen per run
 * -- a saved configuration reaches this step directly, having skipped the rest.
 */
export const EvaluationStep: FC = () => {
  useMessagesBinding();

  return (
    <Stack gap="density-2xl" className="w-full min-w-0">
      <JudgeModelSelect<EvaluationFormValues>
        allowAdapters
        formFieldName="model"
        slotLabel="Model to Evaluate"
        placeholder="Select a model or adapter"
      />

      <Panel elevation="higher" slotHeading="Live Test (optional)" className="min-w-0">
        <Stack gap="density-sm" className="min-w-0">
          <Text kind="body/regular/md" className="text-secondary">
            Test one single row from your dataset with this configuration to validate metric scores
            and model performance.
          </Text>
          <LiveTestPanel />
        </Stack>
      </Panel>
    </Stack>
  );
};
