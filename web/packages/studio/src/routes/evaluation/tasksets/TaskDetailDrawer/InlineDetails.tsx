// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EvaluatorTaskDefinition } from '@nemo/sdk/generated/evals/schema';
import { Text } from '@nvidia/foundations-react-core';
import { DetailSection } from '@studio/routes/evaluation/tasksets/TaskDetailDrawer/DetailSection';
import { JsonBlock } from '@studio/routes/evaluation/tasksets/TaskDetailDrawer/JsonBlock';
import type { FC } from 'react';

export const InlineDetails: FC<{ spec: EvaluatorTaskDefinition }> = ({ spec }) => (
  <>
    <DetailSection heading="Intent">
      <Text className="whitespace-pre-wrap">{spec.intent}</Text>
    </DetailSection>
    <DetailSection heading="Inputs">
      <JsonBlock value={spec.inputs ?? {}} />
    </DetailSection>
    <DetailSection heading="Reference">
      {spec.reference ? (
        <JsonBlock value={spec.reference} />
      ) : (
        <Text className="text-subtle">None.</Text>
      )}
    </DetailSection>
  </>
);
