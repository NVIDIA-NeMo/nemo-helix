// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { SavedTemplateCardProps } from '@studio/components/CreateCustomizationStart/types';
import { SelectableCard } from '@studio/components/SelectableCard';
import { Trash2 } from 'lucide-react';
import type { FC } from 'react';

/**
 * One saved-template tile.
 *
 * Delete sits *beside* the card rather than inside it: `SelectableCard` renders a
 * `<button>`, and a nested button is invalid HTML — the inner click target's behaviour
 * across browsers and screen readers is undefined. An absolutely-positioned sibling keeps
 * both controls independently reachable.
 */
export const SavedTemplateCard: FC<SavedTemplateCardProps> = ({
  template,
  selected,
  onSelect,
  disabled,
  onDelete,
  isDeleting,
}) => (
  <div className="relative">
    <SelectableCard
      selected={selected}
      onSelect={onSelect}
      disabled={disabled}
      className="h-[184px] gap-2 overflow-hidden p-4 pr-12"
    >
      <Text kind="label/regular/sm" className="text-secondary">
        Saved template
      </Text>

      <Stack gap="density-xs" className="min-h-0">
        <Text kind="body/bold/md" className="truncate text-primary" title={template.name}>
          {template.name}
        </Text>
        {template.data.description ? (
          <Text
            kind="body/regular/sm"
            className="line-clamp-2 text-secondary"
            title={template.data.description}
          >
            {template.data.description}
          </Text>
        ) : null}
      </Stack>

      <div className="flex-1" />

      <Flex align="center" gap="density-xs" wrap="wrap">
        <Badge color="gray" kind="outline" size="small">
          {template.data.backend}
        </Badge>
      </Flex>
    </SelectableCard>

    <button
      type="button"
      disabled={isDeleting}
      title={`Delete template ${template.name}`}
      aria-label={`Delete template ${template.name}`}
      onClick={onDelete}
      className="absolute right-2 top-2 rounded p-1.5 text-placeholder transition hover:bg-surface-hover hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#76b900] disabled:cursor-not-allowed disabled:opacity-50"
    >
      <Trash2 size={14} aria-hidden />
    </button>
  </div>
);
