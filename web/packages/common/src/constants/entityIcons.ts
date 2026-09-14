// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Bot,
  Boxes,
  ChartBar,
  Database,
  DatabaseCheck,
  FileStack,
  FlaskConical,
  FolderOpen,
  Form,
  GitCompare,
  Lightbulb,
  ListChecks,
  ListTree,
  LockKeyhole,
  Logs,
  Metronome,
  Radar,
  Rocket,
  ShieldCheck,
  SlidersHorizontal,
  UserPen,
  UsersRound,
  Waypoints,
  type LucideIcon,
} from 'lucide-react';

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
  agents: Bot,
  // Monitor runs are agent telemetry, so they share the span glyph.
  agentMonitorRuns: Logs,
  // Every run that tunes an agent — parameter sweeps, insight loops.
  agentOptimizations: SlidersHorizontal,

  // Models
  baseModels: Boxes,
  customModels: Metronome,
  deployments: Rocket,
  virtualModels: Waypoints,
  inferenceProviders: Radar,

  // Data
  datasets: Database,
  filesets: FileStack,
  filesetFiles: FolderOpen,
  anonymizerJobs: UserPen,
  dataDesignerJobs: Form,
  safeSynthesizerJobs: DatabaseCheck,

  // Governance. A config and its tests are one thing to the user, so the tests
  // take the config's glyph rather than a second shield.
  guardrails: ShieldCheck,
  guardrailChecks: ShieldCheck,

  // Evaluation
  evaluationResults: ChartBar,
  agentEvaluations: ChartBar,
  evaluationSessions: ChartBar,
  experiments: FlaskConical,
  insightExperiments: FlaskConical,
  evalComparison: GitCompare,

  // Observability
  optimizerInsights: Lightbulb,
  telemetryTraces: ListTree,
  insightTraces: ListTree,
  telemetrySpans: Logs,

  // System
  jobs: ListChecks,
  secrets: LockKeyhole,
  members: UsersRound,
} as const satisfies Record<string, LucideIcon>;

/** Every entity with a canonical icon. A superset of the empty-state registry. */
export type EntityKey = keyof typeof ENTITY_ICONS;
