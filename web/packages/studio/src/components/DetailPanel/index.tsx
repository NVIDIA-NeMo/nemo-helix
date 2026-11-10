// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  AccordionContent,
  AccordionItem,
  AccordionRoot,
  AccordionTrigger,
  Flex,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import type { FC, ReactNode } from 'react';

interface DetailPanelProps {
  title: string;
  /** Rendered on the right of the header (e.g. a count or a "View all" link). */
  slotAction?: ReactNode;
  /** When true, the body has no padding so rows can own their own dividers. */
  flush?: boolean;
  /** When true, the header toggles the body, which starts hidden. */
  defaultCollapsed?: boolean;
  children: ReactNode;
}

const PANEL_CLASS = 'w-full rounded-xl border border-base bg-surface-raised';

/** Bordered raised-surface card with a titled header, matching the overview panels. */
export const DetailPanel: FC<DetailPanelProps> = ({
  title,
  slotAction,
  flush,
  defaultCollapsed,
  children,
}) => {
  const header = (
    <>
      <Text kind="body/bold/sm">{title}</Text>
      {slotAction}
    </>
  );
  const body = (
    <>
      <div className="h-px w-full bg-base" />
      <div className={flush ? '' : 'p-4'}>{children}</div>
    </>
  );

  if (defaultCollapsed) {
    return (
      <AccordionRoot collapsible className={`${PANEL_CLASS} overflow-hidden`}>
        <AccordionItem value={title} className="border-b-0">
          <AccordionTrigger className="px-4 py-3.5">{header}</AccordionTrigger>
          <AccordionContent className="p-0">{body}</AccordionContent>
        </AccordionItem>
      </AccordionRoot>
    );
  }

  return (
    <Stack className={PANEL_CLASS}>
      <Flex align="center" justify="between" className="px-4 py-3.5">
        {header}
      </Flex>
      {body}
    </Stack>
  );
};
