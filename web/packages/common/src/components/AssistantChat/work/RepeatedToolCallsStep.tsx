// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TOOL_RUNNING_ACTIONS } from '@nemo/common/src/components/AssistantChat/toolCall/constants';
import { getSubtleToolIcon } from '@nemo/common/src/components/AssistantChat/toolCall/helpers';
import { WORK_ICON_SIZE, WorkStep } from '@nemo/common/src/components/AssistantChat/work/WorkStep';
import { useState, type ReactNode } from 'react';

interface RepeatedToolCallsStepProps {
  readonly children: ReactNode;
  readonly running?: boolean;
  readonly title: string;
  readonly toolName: string;
}

export const RepeatedToolCallsStep = ({
  children,
  running = false,
  title,
  toolName,
}: RepeatedToolCallsStepProps) => {
  const [open, setOpen] = useState(running);
  const [wasRunning, setWasRunning] = useState(running);
  if (running !== wasRunning) {
    setWasRunning(running);
    if (running) {
      setOpen(true);
    }
  }

  const Icon = getSubtleToolIcon(toolName);
  return (
    <WorkStep
      contentClassName="pl-density-xl"
      data-testid="assistant-work-repeated-tool-calls"
      icon={<Icon size={WORK_ICON_SIZE} />}
      open={open}
      onOpenChange={setOpen}
      title={running ? `${TOOL_RUNNING_ACTIONS[toolName] ?? 'Running'}…` : title}
    >
      {children}
    </WorkStep>
  );
};
