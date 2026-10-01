// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { BadgeProps } from '@nvidia/foundations-react-core';
import type { LucideIcon } from 'lucide-react';
import type { ReactNode } from 'react';

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

export interface StartPageProps {
  heading: string;
  headingDescription: string;
  options: StartOption[];
  /** The selected option id. Templates are chosen separately, inside the detail panel. */
  value: string | null;
  onChange: (value: string) => void;
  /** Locks the options and the panel. Set while a selection is being acted on. */
  disabled?: boolean;
  /** The selected option's own panel, rendered under the cards. */
  slotDetail?: ReactNode;
  continueLabel?: ReactNode;
  continueLoading?: boolean;
  canContinue: boolean;
  onContinue: () => void;
  /** Says what is still missing, next to a disabled Continue. */
  blockedHint?: string;
  /** Actions on the current selection, rendered at the start of the footer. */
  slotFooterStart?: ReactNode;
  /** Rendered above the cards, where it stays in view — an error banner, typically. */
  slotBanner?: ReactNode;
}

export interface TemplateGroupsProps {
  groups: StartTemplateGroup[];
  /** The selected template id, independent of which option is selected. */
  value: string | null;
  onChange: (value: string) => void;
  disabled?: boolean;
}
