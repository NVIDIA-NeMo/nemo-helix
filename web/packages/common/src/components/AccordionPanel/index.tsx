// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  PanelContent,
  PanelHeader,
  PanelHeading,
  PanelIcon,
  PanelRoot,
} from '@nvidia/foundations-react-core';
import { ChevronDown, ChevronUp } from '@nvidia/react-gui-icons/icons';
import {
  type ComponentProps,
  type FC,
  type KeyboardEvent,
  type PropsWithChildren,
  type ReactNode,
  useState,
} from 'react';

export interface AccordionPanelProps {
  /** Header text/content — rendered with the same typography as a Panel heading. */
  slotHeading: ReactNode;
  /** Optional leading icon in the header. */
  slotIcon?: ReactNode;
  /** Whether the panel starts expanded. Defaults to collapsed. */
  defaultOpen?: boolean;
  elevation?: ComponentProps<typeof PanelRoot>['elevation'];
  density?: ComponentProps<typeof PanelRoot>['density'];
  className?: string;
  contentClassName?: string;
}

/**
 * A Panel that collapses. Composes the Foundations Panel parts so it is visually
 * identical to a `<Panel>` (border, radius, elevation, heading font), but the
 * header is a toggle with a chevron and the body collapses. Collapsed content is
 * unmounted, so any data fetching inside stops until it's expanded.
 */
export const AccordionPanel: FC<PropsWithChildren<AccordionPanelProps>> = ({
  slotHeading,
  slotIcon,
  defaultOpen = false,
  elevation = 'high',
  density = 'compact',
  className,
  contentClassName,
  children,
}) => {
  const [open, setOpen] = useState(defaultOpen);
  const toggle = () => setOpen((prev) => !prev);
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      toggle();
    }
  };

  return (
    <PanelRoot elevation={elevation} density={density} className={className}>
      <PanelHeader
        role="button"
        tabIndex={0}
        aria-expanded={open}
        onClick={toggle}
        onKeyDown={onKeyDown}
        className="cursor-pointer"
      >
        {slotIcon && <PanelIcon>{slotIcon}</PanelIcon>}
        <PanelHeading>{slotHeading}</PanelHeading>
        <PanelIcon className="ml-auto">{open ? <ChevronUp /> : <ChevronDown />}</PanelIcon>
      </PanelHeader>
      {open && <PanelContent className={contentClassName}>{children}</PanelContent>}
    </PanelRoot>
  );
};
