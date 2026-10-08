// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { BadgeProps, Flex } from '@nvidia/foundations-react-core';
import type { LucideIcon } from 'lucide-react';
import type { ComponentProps, ReactNode } from 'react';

export interface StartOptionTag {
  label: string;
  color: NonNullable<BadgeProps['color']>;
  kind: NonNullable<BadgeProps['kind']>;
}

/**
 * One tile in a "How do you want to start?" row. Generic over the id so each create
 * flow can pin its own union of entry points while sharing the card that renders them.
 */
export interface StartOption<Id extends string = string> {
  id: Id;
  title: string;
  description: string;
  icon: LucideIcon;
  tag?: StartOptionTag;
  /**
   * Whether this option is wired up. Disabled options still render (so the full set
   * of future entry points is visible) but are no-ops — they cannot be selected and
   * never reveal a detail panel or the Continue footer.
   */
  enabled: boolean;
}

/** One template tile below the divider. */
export interface StartTemplate {
  id: string;
  name: string;
  description: string;
  icon: LucideIcon;
  /** A control for this template, laid over its card's end rather than inside the card. */
  action?: ReactNode;
}

export interface StartTemplateGroup {
  id: string;
  title: string;
  templates: StartTemplate[];
  /** Holds the section's place while it fetches; an empty group is dropped instead. */
  loading?: boolean;
  /** Icon colour for this group's tiles. What it signifies is the caller's to decide. */
  accent?: string;
}

export interface StartOptionCardsProps {
  options: StartOption[];
  /** The selected option id. Templates are chosen separately, inside the detail panel. */
  value: string | null;
  onChange: (value: string) => void;
  /** Locks the options and the panel. Set while a selection is being acted on. */
  disabled?: boolean;
}

export interface StartFooterProps {
  continueLabel?: ReactNode;
  continueLoading?: boolean;
  canContinue: boolean;
  onContinue: () => void;
  /** Says what is still missing, next to a disabled Continue. */
  blockedHint?: string;
  /** Actions on the current selection, rendered at the start of the footer. */
  slotFooterStart?: ReactNode;
  attributes?: {
    FlexContainer?: ComponentProps<typeof Flex>;
  };
}

export interface StartPageProps extends StartOptionCardsProps {
  heading: string;
  headingDescription: string;
  /** The selected option's own panel, rendered under the cards. */
  slotDetail?: ReactNode;
  /** Rendered above the cards, where it stays in view — an error banner, typically. */
  slotBanner?: ReactNode;
}

/** A start option's own step, reached from its card, with Back and Continue. */
export interface StartSubPageProps extends Pick<
  StartFooterProps,
  'canContinue' | 'onContinue' | 'blockedHint'
> {
  heading: string;
  headingDescription: string;
  onBack: () => void;
  children: ReactNode;
}

export interface TemplateGroupsProps {
  groups: StartTemplateGroup[];
  onSelect: (id: string) => void;
  /** The template being acted on; its card shows `pendingLabel` while the rest wait. */
  pendingId?: string | null;
  pendingLabel?: string;
  disabled?: boolean;
}
