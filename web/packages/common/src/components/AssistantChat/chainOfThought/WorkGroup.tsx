// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAuiState } from '@assistant-ui/react';
import {
  countToolCalls,
  getWorkDurations,
  getWorkGroupLabel,
  getWorkGroupOrdinal,
} from '@nemo/common/src/components/AssistantChat/chainOfThought/workParts';
import {
  WORK_ICON_SIZE,
  WorkAccordion,
} from '@nemo/common/src/components/AssistantChat/chainOfThought/WorkStep';
import { Lightbulb, WindowCode } from '@nemo/icons/icons';
import { Stack } from '@nvidia/foundations-react-core';
import { type ReactNode, useRef, useState } from 'react';

interface WorkGroupViewProps {
  readonly children: ReactNode;
  readonly defaultOpen?: boolean;
  readonly hasToolCalls: boolean;
  readonly label: string;
}

/** The collapsed "Ran 3 tool calls · 12s" group that holds a reply's thinking and tool calls. */
export const WorkGroupView = ({
  children,
  defaultOpen = false,
  hasToolCalls,
  label,
}: WorkGroupViewProps) => {
  const [open, setOpen] = useState(defaultOpen);
  const triggerRef = useRef<HTMLElement>(null);
  const collapse = () => {
    setOpen(false);
    // The line disappears with the steps, so hand focus back to the group header.
    triggerRef.current?.focus();
  };

  return (
    <WorkAccordion
      contentClassName="p-0"
      data-testid="assistant-work-group"
      icon={
        hasToolCalls ? <WindowCode size={WORK_ICON_SIZE} /> : <Lightbulb size={WORK_ICON_SIZE} />
      }
      onOpenChange={setOpen}
      open={open}
      title={label}
      titleTestId="assistant-work-group-label"
      triggerRef={triggerRef}
    >
      <Stack className="relative ml-1.5 pl-3" data-testid="assistant-work-group-steps">
        {/* The 1px guide line doubles as a collapse control, with a 12px-wide hit area. */}
        <button
          type="button"
          aria-label={`Collapse ${label}`}
          className="group/line absolute inset-y-0 -left-1.5 flex w-3 cursor-pointer justify-center rounded-sm bg-transparent p-0 focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--border-color-interaction-strong)]"
          data-testid="assistant-work-group-collapse"
          onClick={collapse}
          title="Collapse"
        >
          <span className="h-full w-px bg-[var(--border-color-base)] transition-colors group-hover/line:bg-[var(--border-color-interaction-strong)] group-focus-visible/line:bg-[var(--border-color-interaction-strong)]" />
        </button>
        {children}
      </Stack>
    </WorkAccordion>
  );
};

interface WorkGroupProps {
  readonly children: ReactNode;
  /** Indices of the message parts in this group, as reported by `MessagePrimitive.GroupedParts`. */
  readonly indices: readonly number[];
}

/** A `WorkGroupView` labelled from the current message: live while running, then count + time. */
export const WorkGroup = ({ children, indices }: WorkGroupProps) => {
  const startIndex = indices[0] ?? 0;
  const lastIndex = indices.at(-1) ?? startIndex;
  const label = useAuiState(({ message }) => {
    const running = message.status?.type === 'running' && lastIndex === message.parts.length - 1;
    const durationMs = getWorkDurations(message.metadata.custom)[
      getWorkGroupOrdinal(message.parts, startIndex)
    ];
    const toolCallCount = indices.reduce(
      (total, index) => total + (message.parts[index] ? countToolCalls(message.parts[index]) : 0),
      0
    );
    return getWorkGroupLabel({ durationMs, running, toolCallCount });
  });
  const hasToolCalls = useAuiState(({ message }) =>
    indices.some((index) => (message.parts[index] ? countToolCalls(message.parts[index]) : 0) > 0)
  );

  return (
    <WorkGroupView hasToolCalls={hasToolCalls} label={label}>
      {children}
    </WorkGroupView>
  );
};
