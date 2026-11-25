// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AssistantSkill } from '@studio/routes/agents/AssistantChatRoute/types';
import {
  getSkillDisplayName,
  getSkillLookupKeys,
} from '@studio/routes/DashboardLandingRoute/skillDisplayName';

const skill = (overrides: Partial<AssistantSkill>): AssistantSkill => ({
  name: 'inference',
  claude_name: 'nemo-inference',
  description: 'Use NeMo Helix inference.',
  source: 'nemo-helix',
  install_path: '.claude/skills/nemo-inference/SKILL.md',
  installed: false,
  ...overrides,
});

describe('getSkillDisplayName', () => {
  it('title-cases simple skill names', () => {
    expect(getSkillDisplayName(skill({ name: 'inference' }))).toBe('Inference');
  });

  it('strips repeated nemo- prefixes before title-casing', () => {
    expect(getSkillDisplayName(skill({ name: 'nemo-guardrails' }))).toBe('Guardrails');
    expect(getSkillDisplayName(skill({ name: 'nemo-build-agent' }))).toBe('Build Agent');
  });
});

describe('getSkillLookupKeys', () => {
  it('normalizes claude install names back to template keys', () => {
    expect(
      getSkillLookupKeys(skill({ name: 'nemo-guardrails', claude_name: 'nemo-nemo-guardrails' }))
    ).toEqual(expect.arrayContaining(['nemo-guardrails', 'guardrails', 'nemo-nemo-guardrails']));
  });

  it('maps nemo-inference to the inference template key', () => {
    expect(getSkillLookupKeys(skill({ name: 'inference', claude_name: 'nemo-inference' }))).toEqual(
      expect.arrayContaining(['inference', 'nemo-inference'])
    );
  });
});
