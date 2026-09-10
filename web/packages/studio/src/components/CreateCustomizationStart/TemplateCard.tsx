// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { TemplateCardProps } from '@studio/components/CreateCustomizationStart/types';
import { KeyRound } from 'lucide-react';
import type { FC } from 'react';

/** One recipe tile. */
export const TemplateCard: FC<TemplateCardProps> = ({ template, selected, onSelect }) => {
  const requiresHfToken = template.models.some((model) => model.requiresHfToken);
  const stats = `${template.stats.totalParams} params · ${template.stats.activeParams} active · ${template.stats.gpus} GPUs`;

  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      className={`flex h-[184px] w-full flex-col items-start gap-2 overflow-hidden rounded-md border bg-surface-raised p-4 text-left transition focus-visible:border-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#76b900] ${
        selected
          ? 'cursor-pointer border-[#76b900]'
          : 'cursor-pointer border-base hover:-translate-y-0.5 hover:border-[#76b900] hover:bg-surface-hover hover:shadow-md'
      }`}
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
    </button>
  );
};
