// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  AccordionContent,
  AccordionItem,
  AccordionRoot,
  AccordionTrigger,
  Flex,
} from '@nvidia/foundations-react-core';
import type { ReactNode, Ref } from 'react';

export const WORK_ICON_SIZE = '12';

// The work group reads as quiet rows on the canvas rather than KUI's default panel: no fill,
// hover background, or dividers, regular-weight labels that brighten when open, and no horizontal
// padding so rows line up with the edges of the reply. KUI shrinks nested accordion labels to
// 12px; the design keeps every row at 14px.
export const WORK_ACCORDION_CLASS = {
  root: 'w-full min-w-0 bg-transparent [--nv-accordion-label-font-size:var(--text-14)] [--nv-accordion-trigger-bg-hover:transparent] [--nv-accordion-trigger-padding:calc(var(--spacing)*3)_0]',
  item: 'border-b-0',
  trigger: 'font-normal text-secondary hover:text-primary [[open]>&]:text-primary',
  content: 'px-0 pb-2 pt-0',
} as const;

interface WorkAccordionProps {
  readonly children: ReactNode;
  readonly contentClassName?: string;
  readonly 'data-testid'?: string;
  readonly defaultOpen?: boolean;
  readonly icon: ReactNode;
  /** Controlled open state; pair with `onOpenChange`. Uncontrolled (`defaultOpen`) when omitted. */
  readonly open?: boolean;
  readonly onOpenChange?: (open: boolean) => void;
  readonly title: ReactNode;
  readonly titleTestId?: string;
  readonly triggerRef?: Ref<HTMLElement>;
}

/** A single collapsible row with a leading chevron, an icon, and a title. */
export const WorkAccordion = ({
  children,
  contentClassName = WORK_ACCORDION_CLASS.content,
  'data-testid': testId,
  defaultOpen = false,
  icon,
  onOpenChange,
  open,
  title,
  titleTestId,
  triggerRef,
}: WorkAccordionProps) => (
  <AccordionRoot
    className={WORK_ACCORDION_CLASS.root}
    data-testid={testId}
    {...(open === undefined
      ? { defaultValue: defaultOpen ? 'open' : undefined }
      : { value: open ? 'open' : '', onValueChange: (value) => onOpenChange?.(value === 'open') })}
  >
    <AccordionItem value="open" className={WORK_ACCORDION_CLASS.item}>
      <AccordionTrigger
        ref={triggerRef}
        chevronPosition="start"
        className={WORK_ACCORDION_CLASS.trigger}
      >
        <Flex align="center" gap="inherit" className="min-w-0">
          <span className="flex shrink-0 items-center">{icon}</span>
          <span className="min-w-0 truncate" data-testid={titleTestId}>
            {title}
          </span>
        </Flex>
      </AccordionTrigger>
      <AccordionContent className={contentClassName}>{children}</AccordionContent>
    </AccordionItem>
  </AccordionRoot>
);

/** One step in a reply's work group: a Thought, a tool call, or a file edit. */
export const WorkStep = WorkAccordion;
