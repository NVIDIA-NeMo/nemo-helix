// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Cube, Trash } from '@nemo/icons/icons';
import { Badge, Button, Flex, Stack, Tag, Text } from '@nvidia/foundations-react-core';
import { CardIconBadge } from '@studio/components/common/SelectableCard';
import type { BuilderColumn } from '@studio/routes/DataDesignerJobBuildRoute/columns';
import { describeColumn } from '@studio/routes/DataDesignerJobBuildRoute/describeColumn';
import { type FC, memo } from 'react';

/** Accent color → NVIDIA Foundations text token, matching the DAG node icon styling. */
const ACCENT_ICON_CLASS: Record<string, string> = {
  blue: 'text-[color:var(--text-color-accent-blue)]',
  gray: 'text-[color:var(--text-color-accent-gray)]',
  green: 'text-[color:var(--text-color-accent-green)]',
  purple: 'text-[color:var(--text-color-accent-purple)]',
  red: 'text-[color:var(--text-color-accent-red)]',
  teal: 'text-[color:var(--text-color-accent-teal)]',
  yellow: 'text-[color:var(--text-color-accent-yellow)]',
};

export interface SchemaRowProps {
  column: BuilderColumn;
  /** Names of columns this one references, shown inline as `{{ name }}` relationship tags. */
  references: string[];
  selected: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

/**
 * One column rendered as a row in the schema list: a generation-step number, an icon badge,
 * the column name, a type badge, a one-line summary, and its relationship tags. Selecting the
 * row opens the same config pane the DAG canvas uses; the trailing button deletes the column.
 */
export const SchemaRow: FC<SchemaRowProps> = memo(function SchemaRow({
  column,
  references,
  selected,
  onSelect,
  onDelete,
}) {
  const { option } = column;
  const { typeLabel, detail } = describeColumn(column);
  const Icon = option.icon ?? Cube;

  return (
    <Flex
      align="stretch"
      className={`group overflow-hidden rounded-md border bg-surface-raised transition-colors has-[[data-select]:focus-visible]:ring-2 has-[[data-select]:focus-visible]:ring-(--color-brand,#76b900) ${
        selected ? 'border-strong' : 'border-base hover:border-strong'
      }`}
    >
      <button
        type="button"
        data-select=""
        onClick={() => onSelect(column.id)}
        aria-pressed={selected}
        className="flex min-w-0 flex-1 items-center gap-density-md px-density-lg py-density-md text-left focus-visible:outline-none cursor-pointer"
      >
        <CardIconBadge>
          <Icon size="16" className={ACCENT_ICON_CLASS[option.color] ?? ''} aria-hidden />
        </CardIconBadge>

        <Stack gap="density-xxs" className="min-w-0 flex-1">
          <Flex align="center" gap="density-sm" className="min-w-0">
            <Text kind="body/semibold/sm" className="truncate text-primary">
              {column.name || option.label}
            </Text>
            <Tag color={option.color} kind="outline" density="compact" readOnly>
              {typeLabel}
            </Tag>
          </Flex>
          {detail ? (
            <Text kind="body/regular/xs" className="truncate text-secondary">
              {detail}
            </Text>
          ) : null}
          {references.length > 0 ? (
            <Flex wrap="wrap" gap="density-xs" className="mt-density-xxs">
              {references.map((name) => (
                <Badge key={name} color="blue" kind="solid" className="text-[10px]">
                  {`{{${name}}}`}
                </Badge>
              ))}
            </Flex>
          ) : null}
        </Stack>
      </button>

      <Flex align="center" className="shrink-0 border-l border-base">
        <Button
          kind="tertiary"
          color="danger"
          size="tiny"
          onClick={() => onDelete(column.id)}
          aria-label={`Delete ${column.name || option.label}`}
          className="h-full rounded-none opacity-0 transition-opacity focus-visible:opacity-100 group-hover:opacity-100"
        >
          <Trash size="16" aria-hidden />
        </Button>
      </Flex>
    </Flex>
  );
});
