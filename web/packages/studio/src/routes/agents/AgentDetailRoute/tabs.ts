// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  AGENT_OPTIMIZATIONS_ENABLED,
  AGENT_OVERVIEW_ENABLED,
  OPTIMIZER_ENABLED,
} from '@studio/constants/environment';

export const TAB_SEARCH_PARAM = 'tab';

export const DETAIL_TABS = [
  'overview',
  'deployments',
  'logs',
  'chat',
  'evaluations',
  'optimizations',
  'insights',
  'details',
] as const;

export type AgentDetailTab = (typeof DETAIL_TABS)[number];

export const DEFAULT_TAB: AgentDetailTab = AGENT_OVERVIEW_ENABLED ? 'overview' : 'deployments';

/**
 * Opens a modal on arrival, for callers elsewhere in Studio that link to an action rather than
 * a tab. One-shot: the route strips it once handled, so reload and Back do not reopen the modal.
 */
export const ACTION_SEARCH_PARAM = 'action';

export const isAgentDetailTab = (value: string | null): value is AgentDetailTab =>
  !!value &&
  DETAIL_TABS.includes(value as AgentDetailTab) &&
  (value !== 'overview' || AGENT_OVERVIEW_ENABLED) &&
  (value !== 'optimizations' || AGENT_OPTIMIZATIONS_ENABLED) &&
  (value !== 'insights' || OPTIMIZER_ENABLED);
