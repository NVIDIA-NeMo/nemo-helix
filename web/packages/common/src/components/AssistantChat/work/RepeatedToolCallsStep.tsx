// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getSubtleToolIcon } from '@nemo/common/src/components/AssistantChat/toolCall/helpers';
import { WORK_ICON_SIZE, WorkStep } from '@nemo/common/src/components/AssistantChat/work/WorkStep';
import type { ReactNode } from 'react';

interface RepeatedToolCallsStepProps {
  readonly children: ReactNode;
  readonly defaultOpen?: boolean;
  readonly title: string;
  readonly toolName: string;
}

export const RepeatedToolCallsStep = ({
  children,
  defaultOpen,
  title,
  toolName,
}: RepeatedToolCallsStepProps) => {
  const Icon = getSubtleToolIcon(toolName);
  return (
    <WorkStep
      contentClassName="pl-density-xl"
      data-testid="assistant-work-repeated-tool-calls"
      defaultOpen={defaultOpen}
      icon={<Icon size={WORK_ICON_SIZE} />}
      title={title}
    >
      {children}
    </WorkStep>
  );
};
