// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Collapsible, Text } from '@nvidia/foundations-react-core';
import cn from 'classnames';
import type { ReactNode, Ref } from 'react';

interface WorkStepProps {
  readonly children: ReactNode;
  readonly contentClassName?: string;
  readonly 'data-testid'?: string;
  readonly defaultOpen?: boolean;
  readonly icon?: ReactNode;
  readonly open?: boolean;
  readonly onOpenChange?: (open: boolean) => void;
  readonly title: ReactNode;
  readonly titleTestId?: string;
  readonly triggerRef?: Ref<HTMLElement>;
}

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
  <Collapsible
    className="w-full min-w-0"
    data-testid={testId}
    defaultOpen={defaultOpen}
    open={open}
    onOpenChange={onOpenChange}
    showChevron
    slotTrigger={
      <>
        {icon ? <span className="flex shrink-0 items-center">{icon}</span> : null}
        <Text asChild kind="body/regular/md" className="min-w-0 truncate">
          <span data-testid={titleTestId}>{title}</span>
        </Text>
      </>
    }
    attributes={{
      CollapsibleTrigger: {
        ref: triggerRef,
        className:
          'group/step flex w-full cursor-pointer items-center gap-density-sm py-density-sm text-secondary hover:text-primary data-[state=open]:text-primary',
      },
      CollapsibleContent: { className: cn('pb-density-sm', contentClassName) },
    }}
  >
    {children}
  </Collapsible>
);
