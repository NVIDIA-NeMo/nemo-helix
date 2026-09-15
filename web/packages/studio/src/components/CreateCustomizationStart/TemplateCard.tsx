// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { TemplateCardProps } from '@studio/components/CreateCustomizationStart/types';
import { SelectableCard } from '@studio/components/SelectableCard';
import { KeyRound } from 'lucide-react';
import type { FC } from 'react';

/** One recipe tile. */
export const TemplateCard: FC<TemplateCardProps> = ({ template, selected, onSelect }) => {
  const requiresHfToken = template.models.some((model) => model.requiresHfToken);
  const stats = `${template.stats.totalParams} params · ${template.stats.activeParams} active · ${template.stats.gpus} GPUs`;

  return (
    <SelectableCard
      selected={selected}
      onSelect={onSelect}
      className="h-[184px] gap-2 overflow-hidden p-4"
    >
      <Text kind="label/regular/sm" className="text-secondary">
        {template.publisher}
      </Text>

      <Stack gap="density-xs" className="min-h-0">
        <Text kind="body/bold/md" className="text-primary">
          {template.title}
        </Text>
        <Text
          kind="body/regular/sm"
          className="line-clamp-2 text-secondary"
          title={template.description}
        >
          {template.description}
        </Text>
      </Stack>

      <div className="flex-1" />

      <Flex align="center" gap="density-xs" wrap="wrap">
        <Badge color="gray" kind="outline" size="small">
          {template.trainingLabel}
        </Badge>
        {requiresHfToken ? (
          <Flex align="center" gap="density-xs" className="text-placeholder">
            <KeyRound size={12} aria-hidden />
            <Text kind="label/regular/sm">HF token</Text>
          </Flex>
        ) : null}
      </Flex>

      <Text kind="label/regular/sm" className="truncate text-placeholder" title={stats}>
        {stats}
      </Text>
    </SelectableCard>
  );
};
