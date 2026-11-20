// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Mirrors nemo_user_sim_plugin/report.py so the CLI and Studio rank segments the same way.
import type { Conversation, Picked, Scores, Tested } from './types';

/** Persona fields arrive snake_cased: `personal_care_aide` reads as `personal care aide`. */
export const words = (value: string | null | undefined) => (value ?? '').replaceAll('_', ' ');

/** UserSim's interaction styles, colored from easiest to hardest to serve. */
export const STYLE_COLORS = {
  cooperative: 'green',
  neutral: 'gray',
  impatient: 'yellow',
  frustrated: 'red',
  confrontational: 'red',
} as const;

export const styleColor = (style: string | null | undefined) =>
  STYLE_COLORS[(style ?? '') as keyof typeof STYLE_COLORS] ?? 'gray';

export const SEGMENTS = [
  ['persona_age_bin', 'Age'],
  ['persona_education_level', 'Education'],
  ['user_interaction_style', 'Interaction style'],
  ['probe_type', 'Probe'],
] as const;

export type Segment = (typeof SEGMENTS)[number][0];

/** What each scored axis failing means, in words a reviewer reads. */
const AXIS_LABELS: Record<string, string> = {
  'agentic.boundary_held': "called a tool it shouldn't have",
  'agentic.blast_radius_contained': 'took an action with too wide a reach',
  'agentic.consequence_disclosure': "didn't explain what the action would change",
  'agentic.confirmation_sought': "didn't ask before acting",
  'agentic.injection_resistance': 'followed instructions planted in a tool result',
  helpfulness: "didn't help enough",
  accuracy: 'got facts wrong',
  coherence: 'hard to follow',
  safety: 'said something unsafe',
  role_adherence: 'stepped out of its role',
  language_appropriateness: 'wrong language or tone',
};

export const axisLabel = (axis: string) =>
  AXIS_LABELS[axis] ?? words(axis.slice(axis.indexOf('.') + 1));

/** A score's comma separated failed axes as labels. */
export const failedAxes = (reason: string | null) => (reason ? reason.split(', ') : []);

export interface Row extends Conversation {
  passed: boolean | null;
  reason: string | null;
  tested: Tested | null;
  why: Record<string, string>;
  picked: Picked | null;
}

export const joinScores = (conversations: Conversation[], scores: Scores | null): Row[] =>
  conversations.map((conversation) => {
    const score = scores?.[conversation.trajectory_id];
    return {
      ...conversation,
      passed: score?.passed ?? null,
      reason: score?.reason ?? null,
      tested: score?.tested ?? null,
      why: score?.why ?? {},
      picked: score?.picked ?? null,
    };
  });

export interface SegmentRate {
  value: string;
  failed: number;
  scored: number;
}

/** The tool-use check a scenario set's own pick label replaces. */
export const PICK_AXIS = 'agentic.boundary_held';

/** A scenario set's name for the pick, with the tool, as in "Picked the CPU tool: train_model_sklearn". */
export const pickText = (picked: Picked) => `${picked.label}: ${picked.tool}`;

/** The failed checks still worth naming beside a pick label, which already covers the tool check. */
export const otherFailures = (row: Row) =>
  failedAxes(row.reason).filter((axis) => !(row.picked && axis === PICK_AXIS));

/**
 * Scored conversations per segment value, highest rate first. ``counts`` says which rows the rate counts:
 * failures by default, or a scenario set's wrong pick.
 */
export const failureRates = (
  rows: Row[],
  segment: Segment,
  counts: (row: Row) => boolean = (row) => !row.passed
): SegmentRate[] => {
  const rates = new Map<string, SegmentRate>();
  for (const row of rows) {
    if (row.passed === null) continue;
    const value = String(row[segment] ?? 'unknown');
    const rate = rates.get(value) ?? { value, failed: 0, scored: 0 };
    rate.scored += 1;
    rate.failed += counts(row) ? 1 : 0;
    rates.set(value, rate);
  }
  return [...rates.values()].sort((a, b) => b.failed / b.scored - a.failed / a.scored);
};
