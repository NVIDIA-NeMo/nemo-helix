// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AGENT_OPTIMIZATIONS_ENABLED } from '@studio/constants/environment';

export const TAB_SEARCH_PARAM = 'tab';

export const DETAIL_TABS = ['summary', 'chat', 'evaluations', 'optimizations', 'logs'] as const;

export type AgentDetailTab = (typeof DETAIL_TABS)[number];

export const DEFAULT_TAB: AgentDetailTab = 'summary';

const MERGED_INTO_SUMMARY = new Set(['overview', 'details', 'deployments']);

/**
 * Opens a modal on arrival, for callers elsewhere in Studio that link to an action rather than
 * a tab. One-shot: the route strips it once handled, so reload and Back do not reopen the modal.
 */
export const ACTION_SEARCH_PARAM = 'action';

export const isAgentDetailTab = (value: string | null): value is AgentDetailTab =>
  !!value &&
  DETAIL_TABS.includes(value as AgentDetailTab) &&
  (value !== 'optimizations' || AGENT_OPTIMIZATIONS_ENABLED);

export const resolveAgentDetailTab = (value: string | null): AgentDetailTab =>
  isAgentDetailTab(value)
    ? value
    : value && MERGED_INTO_SUMMARY.has(value)
      ? 'summary'
      : DEFAULT_TAB;
