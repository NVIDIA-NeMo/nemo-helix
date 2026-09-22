// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';

/**
 * Entities that deliberately share a glyph, and why. Anything not listed here
 * must own its glyph outright — see the reuse test below.
 */
const INTENTIONAL_ALIASES: Record<string, string> = {
  // Traces are one entity whether reached from Intake or from an Insight.
  insightTraces: 'telemetryTraces',
  // Every evaluation surface reads as "an evaluation".
  agentEvaluations: 'evaluationResults',
  evaluationSessions: 'evaluationResults',
  insightExperiments: 'experiments',
  // A config and its tests are one thing to the user.
  guardrailChecks: 'guardrails',
};

describe('ENTITY_ICONS', () => {
  it('gives every empty-state entity an icon', () => {
    for (const entity of Object.keys(ENTITY_EMPTY_STATES)) {
      expect(ENTITY_ICONS, `${entity} has no canonical icon`).toHaveProperty(entity);
    }
  });

  it('maps every entity to a renderable icon', () => {
    for (const [entity, icon] of Object.entries(ENTITY_ICONS)) {
      expect(icon, `${entity} has no icon`).toBeTruthy();
    }
  });

  it('points each alias at its parent entity glyph', () => {
    for (const [alias, parent] of Object.entries(INTENTIONAL_ALIASES)) {
      expect(
        ENTITY_ICONS[alias as keyof typeof ENTITY_ICONS],
        `${alias} should share the ${parent} glyph`
      ).toBe(ENTITY_ICONS[parent as keyof typeof ENTITY_ICONS]);
    }
  });

  it('does not reuse one glyph across unrelated entities', () => {
    // The regression guard for ASTD-447: a shared glyph is how `ShieldCheck`
    // and `Radar` each ended up covering two unrelated entities, at which point
    // neither glyph meant anything specific.
    const owned = Object.entries(ENTITY_ICONS).filter(
      ([entity]) => !(entity in INTENTIONAL_ALIASES)
    );
    const byIcon = new Map<unknown, string[]>();
    for (const [entity, icon] of owned) {
      byIcon.set(icon, [...(byIcon.get(icon) ?? []), entity]);
    }

    const collisions = [...byIcon.values()].filter((entities) => entities.length > 1);
    expect(collisions, 'unrelated entities sharing a glyph').toEqual([]);
  });
});
