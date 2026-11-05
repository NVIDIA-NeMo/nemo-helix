// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  findToolStepDefinition,
  getFallbackToolStep,
  getRepeatedToolStepTitle,
} from '@nemo/common/src/components/AssistantChat/chainOfThought/toolSteps';
import {
  WORK_ACCORDION_CLASS,
  WORK_ICON_SIZE,
  WorkStep,
} from '@nemo/common/src/components/AssistantChat/chainOfThought/WorkStep';
import cn from 'classnames';
import type { ReactNode } from 'react';

interface RepeatedToolCallsStepProps {
  /** The individual calls, usually `ToolCallStep`s, shown indented when the step is open. */
  readonly children: ReactNode;
  readonly count: number;
  readonly defaultOpen?: boolean;
  readonly toolName: string;
}

/** Several calls of the same tool folded into one step, e.g. "Read 3 files". */
export const RepeatedToolCallsStep = ({
  children,
  count,
  defaultOpen,
  toolName,
}: RepeatedToolCallsStepProps) => {
  const step = findToolStepDefinition(toolName) ?? getFallbackToolStep(toolName);
  const { icon: Icon } = step;
  return (
    <WorkStep
      contentClassName={cn(WORK_ACCORDION_CLASS.content, 'pl-6')}
      data-testid="assistant-work-repeated-tool-calls"
      defaultOpen={defaultOpen}
      icon={<Icon size={WORK_ICON_SIZE} />}
      title={getRepeatedToolStepTitle(step, count)}
    >
      {children}
    </WorkStep>
  );
};
