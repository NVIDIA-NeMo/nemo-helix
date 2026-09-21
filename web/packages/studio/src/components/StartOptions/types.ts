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

export interface StartOptionCardProps {
  option: StartOption;
  selected: boolean;
  /** Fired on click / keyboard activation. Only invoked for enabled options. */
  onSelect: () => void;
  /**
   * Lays the icon beside the title instead of above it and drops the fixed height, so the
   * row takes about half the vertical space. For flows where the tiles are a step on the
   * way somewhere rather than the main event.
   */
  compact?: boolean;
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
}

export interface StartPageProps {
  heading: string;
  headingDescription: string;
  options: StartOption[];
  templateGroups?: StartTemplateGroup[];
  /**
   * The selected option or template id. Both live in one radio group: a template is
   * picked outright rather than by first choosing an "and then pick one" option, so the
   * two sets of ids share a namespace and must not collide.
   */
  value: string | null;
  onChange: (value: string) => void;
  /** Locks the whole group. Set while a selection is being acted on. */
  disabled?: boolean;
  /** Shown while nothing is selected, in place of the default prompt. */
  emptyHint?: string;
  continueLabel?: ReactNode;
  continueLoading?: boolean;
  canContinue: boolean;
  onContinue: () => void;
  /** Rendered above the footer — an error banner, typically. */
  slotBanner?: ReactNode;
}
