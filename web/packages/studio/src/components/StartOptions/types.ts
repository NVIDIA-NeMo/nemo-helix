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
  /**
   * Renders the heading over placeholder tiles instead of the group's own. A group whose
   * templates are still being fetched would otherwise be indistinguishable from an empty
   * one, which is dropped — the section would vanish and then push the page down on arrival.
   */
  loading?: boolean;
  /**
   * Colour for this group's tile icons, as a CSS colour or token reference. What it
   * signifies is the caller's to decide; the page only applies it.
   */
  accent?: string;
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
  /**
   * Badge shown on the divider, beside "OR START FROM A TEMPLATE" — the templates'
   * counterpart to the per-option tags.
   */
  templatesTag?: StartOptionTag;
  /** Rendered above the footer — an error banner, typically. */
  slotBanner?: ReactNode;
}
