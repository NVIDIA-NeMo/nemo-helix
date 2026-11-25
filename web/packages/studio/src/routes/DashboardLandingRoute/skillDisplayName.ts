// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AssistantSkill } from '@studio/routes/agents/AssistantChatRoute/types';

const titleCaseSkillSegment = (segment: string): string =>
  segment ? segment.charAt(0).toUpperCase() + segment.slice(1) : segment;

/** Strip repeated ``nemo-`` prefixes before title-casing skill folder names. */
export const getSkillLookupKeys = (skill: AssistantSkill): string[] => {
  const keys = new Set<string>();

  for (const rawName of [skill.name, skill.claude_name]) {
    let current = rawName;
    keys.add(current);
    while (current.startsWith('nemo-')) {
      current = current.slice(5);
      keys.add(current);
    }
  }

  return [...keys];
};

export const getSkillDisplayName = (skill: AssistantSkill): string => {
  let name = skill.name;
  while (name.startsWith('nemo-')) {
    name = name.slice(5);
  }

  const displayName = name.split('-').filter(Boolean).map(titleCaseSkillSegment).join(' ');

  return displayName || skill.claude_name;
};
