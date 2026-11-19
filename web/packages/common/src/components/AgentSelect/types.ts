// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Agent } from '@nemo/sdk/generated/agents/schema';

interface AgentSelectBaseProps {
  /**
   * The loaded page(s) of agents. The dropdown renders exactly what it is given — pair with
   * {@link onSearchChange} and {@link onLoadMore} (see `useAgentSearch`, or just use
   * `WorkspaceAgentSelect`) so a large workspace arrives one page at a time.
   */
  agents?: Agent[];
  /**
   * Heading for each agent; agents sharing one render together under it, in first-seen order.
   * Omit for a flat list. `groupAgentsByFormat` groups by Fabric / NAT workflow.
   */
  groupBy?: (agent: Agent) => string;
  /**
   * Server-side search. When set, the dropdown stops filtering {@link agents} itself and reports
   * the debounced filter text instead; the caller re-queries and passes back new `agents`.
   */
  onSearchChange?: (search: string) => void;
  /** How long the filter box settles before {@link onSearchChange} fires. */
  searchDebounceMs?: number;
  /** Called as the list scrolls near its end. Ignored unless {@link hasMore} is true. */
  onLoadMore?: () => void | Promise<void>;
  /** Whether another page is available for {@link onLoadMore} to fetch. */
  hasMore?: boolean;
  /** Whether the next page is in flight. */
  isLoadingMore?: boolean;
  /** Footer copy shown once every page has loaded. Omit to render nothing. */
  doneLoadingMessage?: string;
  /** Copy shown when no agents match. */
  emptyMessage?: string;
  /** Whether the first page of agents is still loading. */
  loading?: boolean;
  /**
   * Why an agent cannot be picked here, or undefined when it can. Disabled agents stay listed
   * (dimmed) and show the reason in their hover details, rather than silently disappearing.
   */
  getDisabledReason?: (agent: Agent) => string | undefined;
  disabled?: boolean;
  placeholder?: string;
  fullWidth?: boolean;
  dropdownSide?: 'top' | 'bottom';
  onOpenChange?: (open: boolean) => void;
  'aria-label'?: string;
}

export interface SingleAgentSelectProps extends AgentSelectBaseProps {
  multiple?: false;
  /** Selected agent name. */
  value: string | null;
  onValueChange: (name: string, agent: Agent) => void;
}

export interface MultiAgentSelectProps extends AgentSelectBaseProps {
  multiple: true;
  /** Selected agent names, in selection order. */
  value: readonly string[];
  /**
   * Called with the full new selection. `agents` holds the entity for every selected name the
   * dropdown has seen; names restored from elsewhere and never loaded are left out.
   */
  onValueChange: (names: string[], agents: Agent[]) => void;
  /** Cap on how many agents can be selected; further rows are disabled once it is reached. */
  maxSelected?: number;
}

export type AgentSelectProps = SingleAgentSelectProps | MultiAgentSelectProps;

export type AgentRow =
  | { kind: 'heading'; key: string; label: string }
  | { kind: 'agent'; key: string; agent: Agent };
