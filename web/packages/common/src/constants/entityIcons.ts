// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NvidiaGuiIconProps } from '@nvidia/react-gui-icons';
import {
  Robot,
  CubeStack,
  ChartBar,
  Db,
  Layers,
  Beaker,
  FolderOpen,
  Generate,
  ListCheckmark,
  ScmCompare,
  Lightbulb,
  ChartTree,
  LockClosed,
  Pulse,
  Radar,
  Rocket,
  Shield,
  Sliders,
  Profile,
  ProfileGroup,
  Network,
} from '@nvidia/react-gui-icons/icons';
import type { FC } from 'react';

/**
 * The single canonical glyph for every entity Studio represents.
 *
 * One entity means one icon, everywhere it appears — sidebar nav, route header,
 * empty state, primary CTA. Import from here rather than reaching for
 * `lucide-react` at the callsite: a locally chosen glyph is how the surfaces
 * drifted apart in the first place (ASTD-447).
 *
 * Choosing a glyph for a new entity:
 * - Reuse the parent's glyph for a sub-entity the user reads as part of the
 *   parent (`guardrails` / `guardrailChecks`), or the parent's family where the
 *   two are genuinely distinct (`telemetryTraces` / `telemetrySpans`).
 * - Never reuse a glyph already spoken for by an *unrelated* entity. Two
 *   unrelated entities on one glyph means neither one owns it.
 *
 * Not every entity here has an empty state yet — this map is the superset, and
 * `ENTITY_EMPTY_STATES` covers whichever subset has migrated onto
 * {@link EntityEmptyState}.
 */
export const ENTITY_ICONS = {
  // Agents
  agents: Robot,
  // Monitor runs are agent telemetry, so they share the span glyph.
  agentMonitorRuns: Pulse,
  // Every run that tunes an agent — parameter sweeps, insight loops.
  agentOptimizations: Sliders,

  // Models
  baseModels: CubeStack,
  customModels: Sliders,
  deployments: Rocket,
  virtualModels: Network,
  inferenceProviders: Radar,

  // Data
  datasets: Db,
  filesets: Layers,
  filesetFiles: FolderOpen,
  anonymizerJobs: Profile,
  dataDesignerJobs: Generate,
  safeSynthesizerJobs: Db,

  // Governance. A config and its tests are one thing to the user, so the tests
  // take the config's glyph rather than a second shield.
  guardrails: Shield,
  guardrailChecks: Shield,

  // Evaluation
  evaluationResults: ChartBar,
  agentEvaluations: ChartBar,
  evaluationSessions: ChartBar,
  experiments: Beaker,
  insightExperiments: Beaker,
  evalComparison: ScmCompare,

  // Observability
  optimizerInsights: Lightbulb,
  telemetryTraces: ChartTree,
  insightTraces: ChartTree,
  telemetrySpans: Pulse,

  // System
  jobs: ListCheckmark,
  recentJobs: ListCheckmark,
  secrets: LockClosed,
  members: ProfileGroup,
} as const satisfies Record<string, FC<Omit<NvidiaGuiIconProps, 'iconName'>>>;

/** Every entity with a canonical icon. A superset of the empty-state registry. */
export type EntityKey = keyof typeof ENTITY_ICONS;
