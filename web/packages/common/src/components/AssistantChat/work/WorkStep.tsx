// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  CollapsibleContent,
  CollapsibleRoot,
  CollapsibleTrigger,
  Text,
} from '@nvidia/foundations-react-core';
import cn from 'classnames';
import type { ReactNode, Ref } from 'react';

export const WORK_ICON_SIZE = '12';

interface WorkStepProps {
  readonly children: ReactNode;
  readonly contentClassName?: string;
  readonly 'data-testid'?: string;
  readonly defaultOpen?: boolean;
  readonly icon?: ReactNode;
  /** Controlled open state; pair with `onOpenChange`. Uncontrolled (`defaultOpen`) when omitted. */
  readonly open?: boolean;
  readonly onOpenChange?: (open: boolean) => void;
  readonly title: ReactNode;
  readonly titleTestId?: string;
  readonly triggerRef?: Ref<HTMLElement>;
}

/**
 * One collapsible row in a reply's work group (the group header, a Thought, a tool call) with a
 * leading chevron, an icon, and a title. Children of the title can use `group-hover/step` and
 * `group-data-[state=open]/step` to follow the row's hover and open states.
 */
export const WorkStep = ({
  children,
  contentClassName,
  'data-testid': testId,
  defaultOpen = false,
  icon,
  onOpenChange,
  open,
  title,
  titleTestId,
  triggerRef,
}: WorkStepProps) => (
  <CollapsibleRoot
    className="w-full min-w-0"
    data-testid={testId}
    defaultOpen={defaultOpen}
    open={open}
    onOpenChange={onOpenChange}
  >
    <CollapsibleTrigger
      ref={triggerRef}
      showChevron
      className="group/step flex w-full cursor-pointer items-center gap-density-sm py-density-sm text-secondary hover:text-primary data-[state=open]:text-primary"
    >
      {icon ? <span className="flex shrink-0 items-center">{icon}</span> : null}
      <Text asChild kind="body/regular/md" className="min-w-0 truncate">
        <span data-testid={titleTestId}>{title}</span>
      </Text>
    </CollapsibleTrigger>
    <CollapsibleContent className={cn('pb-density-sm', contentClassName)}>
      {children}
    </CollapsibleContent>
  </CollapsibleRoot>
);
