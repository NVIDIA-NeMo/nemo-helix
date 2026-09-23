// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { BadgeProps, Text } from '@nvidia/foundations-react-core';
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
  templateGroups?: StartTemplateGroup[];
  /** Picking a tile runs the flow straight away — there is no confirm step. */
  onSelect: (id: string) => void;
  /** Locks every tile while a pick is being acted on. */
  disabled?: boolean;
  /** The tile being acted on, marked active and given a spinner. */
  busyId?: string | null;
  /** Progress text shown on the busy tile. */
  busyLabel?: string;
  /** Badge on the divider — the templates' counterpart to the per-option tags. */
  templatesTag?: StartOptionTag;
  /** Rendered below the tiles — an error banner, typically. */
  slotBanner?: ReactNode;
}

export interface StartTileProps {
  label: ReactNode;
  description?: ReactNode;
  icon: ReactNode;
  /** Trailing content on the label row, pushed right — a Badge, typically. */
  slotEnd?: ReactNode;
  onSelect: () => void;
  disabled?: boolean;
  /** Swaps `slotEnd` for a spinner and marks the tile active. */
  busy?: boolean;
  busyLabel?: string;
  labelKind?: ComponentProps<typeof Text>['kind'];
  descriptionKind?: ComponentProps<typeof Text>['kind'];
  className?: string;
}
