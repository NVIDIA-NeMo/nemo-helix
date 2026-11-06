// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getSubtleToolIcon } from '@nemo/common/src/components/AssistantChat/toolCall/helpers';
import {
  WORK_ACCORDION_CLASS,
  WORK_ICON_SIZE,
  WorkStep,
} from '@nemo/common/src/components/AssistantChat/work/WorkStep';
import cn from 'classnames';
import type { ReactNode } from 'react';

interface RepeatedToolCallsStepProps {
  /** The individual calls, usually `ToolCallStep`s, shown indented when the step is open. */
  readonly children: ReactNode;
  readonly defaultOpen?: boolean;
  /** The folded message, e.g. "Read 3 files" from `summarizeRepeatedSubtleToolActions`. */
  readonly title: string;
  readonly toolName: string;
}

/** Several calls of the same tool folded into one step. */
export const RepeatedToolCallsStep = ({
  children,
  defaultOpen,
  title,
  toolName,
}: RepeatedToolCallsStepProps) => {
  const Icon = getSubtleToolIcon(toolName);
  return (
    <WorkStep
      contentClassName={cn(WORK_ACCORDION_CLASS.content, 'pl-6')}
      data-testid="assistant-work-repeated-tool-calls"
      defaultOpen={defaultOpen}
      icon={<Icon size={WORK_ICON_SIZE} />}
      title={title}
    >
      {children}
    </WorkStep>
  );
};
