// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { DescribeWithAiPanel } from '@studio/components/CreateFilesetStart/DescribeWithAiPanel';
import { buildTemplateGroups } from '@studio/components/CreateFilesetStart/templateGroups';
import type {
  DetailPoint,
  StartOption,
  StartOptionDetailProps,
} from '@studio/components/CreateFilesetStart/types';
import { TemplateGroups } from '@studio/components/StartOptions/TemplateGroups';
import { Layers, Sparkles, Wand2 } from 'lucide-react';
import { useMemo, type FC, type ReactNode } from 'react';

const SCRATCH_POINTS: DetailPoint[] = [
  {
    icon: Layers,
    title: 'Add columns block by block',
    description:
      'Drop in samplers, LLM generations, transforms and validators in any order on an empty canvas.',
  },
  {
    icon: Wand2,
    title: 'Wire columns together',
    description: 'Reference earlier columns in prompts and expressions to build up each record.',
  },
  {
    icon: Sparkles,
    title: 'Preview and run',
    description: 'Generate a sample at any time, tweak, and run the full job when it looks right.',
  },
];

const DETAIL_CONTENT: Partial<Record<StartOption['id'], ReactNode>> = {
  scratch: (
    <Flex gap="density-md" className="w-full flex-wrap">
      {SCRATCH_POINTS.map(({ icon: Icon, title, description }) => (
        <Stack
          key={title}
          gap="density-xs"
          className="min-w-[260px] flex-1 rounded-md border border-base bg-surface-raised p-5"
        >
          <Flex
            align="center"
            justify="center"
            className="size-8 shrink-0 rounded-md bg-surface-sunken"
          >
            <Icon size={16} className="text-primary" aria-hidden />
          </Flex>
          <Text kind="body/semibold/sm" className="text-primary">
            {title}
          </Text>
          <Text kind="body/regular/sm" className="text-secondary">
            {description}
          </Text>
        </Stack>
      ))}
    </Flex>
  ),
};

export const StartOptionDetail: FC<StartOptionDetailProps> = ({
  option,
  selectedTemplateId,
  onSelectTemplate,
  workspace,
  onValidConfig,
}) => {
  const templateGroups = useMemo(buildTemplateGroups, []);

  let content: ReactNode;
  if (option.id === 'template') {
    content = (
      <TemplateGroups
        groups={templateGroups}
        value={selectedTemplateId}
        onChange={onSelectTemplate}
      />
    );
  } else if (option.id === 'ai') {
    content = <DescribeWithAiPanel workspace={workspace} onValidConfig={onValidConfig} />;
  } else {
    content = DETAIL_CONTENT[option.id];
  }

  if (!content) {
    return null;
  }

  return (
    <Stack gap="density-2xl" className="w-full">
      {content}
    </Stack>
  );
};
