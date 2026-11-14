// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAuiState } from '@assistant-ui/react';
import {
  countGroupToolCalls,
  getWorkedForLabel,
  getWorkGroupOrdinal,
  getWorkGroupLabel,
} from '@nemo/common/src/components/AssistantChat/work/workParts';
import { WORK_ICON_SIZE, WorkStep } from '@nemo/common/src/components/AssistantChat/work/WorkStep';
import { Lightbulb, WindowCode } from '@nemo/icons/icons';
import { Stack } from '@nvidia/foundations-react-core';
import type { ReactNode } from 'react';

interface WorkGroupViewProps {
  readonly children: ReactNode;
  readonly defaultOpen?: boolean;
  readonly hasToolCalls: boolean;
  readonly label: string;
  readonly open?: boolean;
  readonly onOpenChange?: (open: boolean) => void;
}

export const WorkGroupView = ({
  children,
  defaultOpen,
  hasToolCalls,
  label,
  onOpenChange,
  open,
}: WorkGroupViewProps) => (
  <WorkStep
    contentClassName="pb-0"
    data-testid="assistant-work-group"
    defaultOpen={defaultOpen}
    icon={hasToolCalls ? <WindowCode size={WORK_ICON_SIZE} /> : <Lightbulb size={WORK_ICON_SIZE} />}
    onOpenChange={onOpenChange}
    open={open}
    title={label}
    titleTestId="assistant-work-group-label"
  >
    <Stack
      className="ml-density-sm border-l border-base pl-density-md"
      data-testid="assistant-work-group-steps"
    >
      {children}
    </Stack>
  </WorkStep>
);

interface WorkGroupProps {
  readonly children: ReactNode;
  readonly indices: readonly number[];
  readonly onUserOpenChange: (ordinal: number, open: boolean) => void;
  readonly userOpenByOrdinal: ReadonlyMap<number, boolean>;
}

/** Open while the reply runs, then collapsed, unless the reader toggled it in the meantime. */
export const WorkGroup = ({
  children,
  indices,
  onUserOpenChange,
  userOpenByOrdinal,
}: WorkGroupProps) => {
  const firstIndex = indices[0] ?? 0;
  const lastIndex = indices.at(-1) ?? 0;
  const ordinal = useAuiState(({ message }) => getWorkGroupOrdinal(message.parts, firstIndex));
  const running = useAuiState(({ message }) => message.status?.type === 'running');
  const active = useAuiState(
    ({ message }) => message.status?.type === 'running' && lastIndex === message.parts.length - 1
  );
  const toolCallCount = useAuiState(({ message }) => countGroupToolCalls(message.parts, indices));
  const workedFor = useAuiState(({ message }) => getWorkedForLabel(message.parts, indices));

  return (
    <WorkGroupView
      hasToolCalls={toolCallCount > 0}
      label={getWorkGroupLabel({ active, toolCallCount, workedFor })}
      onOpenChange={(open) => onUserOpenChange(ordinal, open)}
      open={userOpenByOrdinal.get(ordinal) ?? running}
    >
      {children}
    </WorkGroupView>
  );
};
