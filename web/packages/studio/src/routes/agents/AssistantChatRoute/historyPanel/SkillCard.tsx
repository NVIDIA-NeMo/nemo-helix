// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Card, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { AssistantSkill } from '@studio/routes/agents/AssistantChatRoute/types';
import { getSkillDisplayName } from '@studio/routes/DashboardLandingRoute/skillDisplayName';
import { BookOpen } from 'lucide-react';

export const SkillCard = ({ skill }: { skill: AssistantSkill }) => (
  <Card
    className="h-auto shadow-none [&_.nv-card-content]:p-density-md"
    attributes={{ CardContent: { className: 'min-h-0' } }}
  >
    <Stack gap="density-sm" className="min-w-0">
      <Flex align="start" gap="density-sm" className="min-w-0">
        <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded bg-surface-sunken text-secondary">
          <BookOpen size={14} />
        </span>
        <Stack gap="density-xxs" className="min-w-0">
          <Text kind="body/semibold/sm" className="truncate" title={skill.name}>
            {getSkillDisplayName(skill)}
          </Text>
          <Text kind="body/regular/xs" className="truncate text-secondary">
            {skill.claude_name}
          </Text>
        </Stack>
      </Flex>
      <Text kind="body/regular/sm" className="break-words text-secondary">
        {skill.description || 'No description'}
      </Text>
    </Stack>
  </Card>
);
