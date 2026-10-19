// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner, Button, Flex, Stack, Text, Tooltip } from '@nvidia/foundations-react-core';
import { Empty } from '@studio/components/Empty';
import {
  ASSISTANT_SKILLS_QUERY_KEY,
  listAssistantSkills,
} from '@studio/routes/agents/AssistantChatRoute/api';
import { SkillsPanelSkeleton } from '@studio/routes/agents/AssistantChatRoute/historyPanel/HistoryPanelSkeletons';
import { SkillCard } from '@studio/routes/agents/AssistantChatRoute/historyPanel/SkillCard';
import { useQuery } from '@tanstack/react-query';
import { RefreshCw } from 'lucide-react';

export const SkillsPanelContents = () => {
  const {
    data: skills = [],
    error,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ASSISTANT_SKILLS_QUERY_KEY,
    queryFn: listAssistantSkills,
  });

  return (
    <>
      <Flex
        align="center"
        justify="between"
        gap="density-sm"
        className="border-b border-base px-density-md py-density-sm"
      >
        <Text className="text-secondary" kind="body/regular/sm">
          {skills.length} skills
        </Text>
        <Tooltip slotContent="Refresh skills">
          <Button
            aria-label="Refresh skills"
            kind="tertiary"
            size="small"
            type="button"
            disabled={isLoading}
            onClick={() => void refetch()}
          >
            <RefreshCw size={16} />
          </Button>
        </Tooltip>
      </Flex>
      {error && (
        <div className="px-density-md py-density-sm">
          <Banner kind="inline" status="error">
            Could not load NeMo Assistant skills.
          </Banner>
        </div>
      )}
      {isLoading ? (
        <SkillsPanelSkeleton />
      ) : skills.length ? (
        <div className="min-h-0 flex-1 overflow-y-auto p-density-sm">
          <Stack gap="density-md">
            {skills.map((skill) => (
              <SkillCard key={skill.claude_name} skill={skill} />
            ))}
          </Stack>
        </div>
      ) : !error ? (
        <Flex className="min-h-0 flex-1" align="center" justify="center">
          <Empty title="No skills found" description="NeMo Assistant skills will appear here." />
        </Flex>
      ) : null}
    </>
  );
};
