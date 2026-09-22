// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Card, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { JudgeModelSelect } from '@studio/components/evaluation/JudgeModelSelect';
import { LiveTestPanel } from '@studio/routes/evaluation/EvaluationNewRoute/LiveTestPanel';
import { type EvaluationFormValues } from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { useMessagesBinding } from '@studio/routes/evaluation/EvaluationNewRoute/useMessagesBinding';
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
        formFieldName="model"
        slotLabel="Model to Evaluate"
        placeholder="Select a model"
      />

      <Card className="min-w-0 p-density-lg">
        <Stack gap="density-sm" className="min-w-0">
          <Flex align="center" gap="density-sm">
            <Text kind="label/bold/xl">Live Test</Text>
            <Text kind="body/regular/md" className="text-secondary">
              Optional
            </Text>
          </Flex>
          <Text kind="body/regular/md" className="text-secondary">
            Test one single row from your dataset with this configuration to validate metric scores
            and model performance.
          </Text>
          <LiveTestPanel />
        </Stack>
      </Card>
    </Stack>
  );
};
