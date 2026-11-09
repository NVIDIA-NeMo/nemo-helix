// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DetailPageTab } from '@studio/components/Layouts/DetailPageLayout';
import {
  AGENT_OPTIMIZATIONS_ENABLED,
  AGENT_OVERVIEW_ENABLED,
  OPTIMIZER_ENABLED,
} from '@studio/constants/environment';

export const TAB_SEARCH_PARAM = 'tab';

export type AgentDetailTab =
  | 'overview'
  | 'chat'
  | 'evaluations'
  | 'insights'
  | 'optimizations'
  | 'deployments'
  | 'logs';

/** The tabs this build shows, in order. Flagged-off tabs are left out, so their URLs fall back to
 *  DEFAULT_TAB. */
export const AGENT_DETAIL_TABS: DetailPageTab<AgentDetailTab>[] = [
  ...(AGENT_OVERVIEW_ENABLED ? [{ value: 'overview' as const, label: 'Overview' }] : []),
  { value: 'chat', label: 'Chat' },
  { value: 'evaluations', label: 'Evaluations' },
  ...(OPTIMIZER_ENABLED ? [{ value: 'insights' as const, label: 'Insights' }] : []),
  ...(AGENT_OPTIMIZATIONS_ENABLED
    ? [{ value: 'optimizations' as const, label: 'Optimizations' }]
    : []),
  { value: 'deployments', label: 'Deployment' },
  { value: 'logs', label: 'Logs' },
];

export const EVAL_TABS: ReadonlySet<AgentDetailTab> = new Set([
  'overview',
  'evaluations',
  'optimizations',
]);

export const DEFAULT_TAB: AgentDetailTab = AGENT_OVERVIEW_ENABLED ? 'overview' : 'deployments';

/**
 * Opens a modal on arrival, for callers elsewhere in Studio that link to an action rather than
 * a tab. One-shot: the route strips it once handled, so reload and Back do not reopen the modal.
 */
export const ACTION_SEARCH_PARAM = 'action';

/**
 * Which view the selected tab is showing, for tabs that have more than one. `tab` already belongs
 * to the outer tabs, so the views get their own parameter. Each tab reads only the values it
 * knows, and absent means its default view.
 */
export const VIEW_SEARCH_PARAM = 'view';

/** Table, then how to set the study up, then — for a guided path — that strategy's form. */
export enum OptimizationView {
  Table = 'table',
  Strategy = 'strategy',
  Form = 'form',
  Routing = 'routing',
}

export const isOptimizationView = (value: string | null): value is OptimizationView =>
  Object.values(OptimizationView).some((view) => view === value);

export const isAgentDetailTab = (value: string | null): value is AgentDetailTab =>
  AGENT_DETAIL_TABS.some((tab) => tab.value === value);
