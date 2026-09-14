// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AGENT_OPTIMIZATIONS_ENABLED, AGENT_OVERVIEW_ENABLED } from '@studio/constants/environment';

export const TAB_SEARCH_PARAM = 'tab';

export const DETAIL_TABS = [
  'overview',
  'deployments',
  'logs',
  'chat',
  'evaluations',
  'optimizations',
  'details',
] as const;

export type AgentDetailTab = (typeof DETAIL_TABS)[number];

export const DEFAULT_TAB: AgentDetailTab = AGENT_OVERVIEW_ENABLED ? 'overview' : 'deployments';

export const isAgentDetailTab = (value: string | null): value is AgentDetailTab =>
  !!value &&
  DETAIL_TABS.includes(value as AgentDetailTab) &&
  (value !== 'overview' || AGENT_OVERVIEW_ENABLED) &&
  (value !== 'optimizations' || AGENT_OPTIMIZATIONS_ENABLED);
