// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MagnifyingGlass } from '@nemo/icons/icons';
import { Stack, Text, TextInput } from '@nvidia/foundations-react-core';
import { ColumnTypeGroupSection } from '@studio/components/AddColumnPalette/ColumnTypeGroupSection';
import { COLUMN_TYPE_GROUPS } from '@studio/components/AddColumnPalette/constants';
import type {
  AddColumnSelection,
  ColumnTypeOption,
} from '@studio/components/AddColumnPalette/types';
import { type FC, useMemo, useState } from 'react';

/** Matches an option against a lowercased search query (name + description). */
const matchesQuery = (option: ColumnTypeOption, query: string): boolean =>
  option.label.toLowerCase().includes(query) || option.description.toLowerCase().includes(query);

export interface AddColumnPaletteProps {
  /** Called with the chosen column type when an option is activated. */
  onAddColumn?: (selection: AddColumnSelection) => void;
  /**
   * Disabled reasons keyed by column type. An option whose column type has an entry renders as a
   * disabled card with the reason as its tooltip — e.g. `{ 'seed-dataset': 'Only one…' }`.
   */
  disabledReasons?: Partial<Record<string, string>>;
  className?: string;
}

/**
 * "Add a column" palette for the Data Designer recipe builder.
 *
 * Lists every Data Designer column type as a keyboard-activatable card, grouped by
 * family (Sampler — broken out into its sub-types — then Generate, Transform, Validate,
 * and Data & custom). A search box filters across names and descriptions. Purely
 * presentational: wire {@link AddColumnPaletteProps.onAddColumn} to append a column to
 * the recipe.
 */
export const AddColumnPalette: FC<AddColumnPaletteProps> = ({
  onAddColumn,
  disabledReasons,
  className,
}) => {
  const [search, setSearch] = useState('');

  const handleSelect = (selection: AddColumnSelection) => onAddColumn?.(selection);

  const filteredGroups = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) {
      return COLUMN_TYPE_GROUPS.map((group) => ({ group, options: group.options }));
    }
    return COLUMN_TYPE_GROUPS.map((group) => ({
      group,
      options: group.options.filter((option) => matchesQuery(option, query)),
    })).filter(({ options }) => options.length > 0);
  }, [search]);

  return (
    <Stack gap="density-lg" className={`flex h-full min-h-0 flex-col ${className ?? ''}`}>
      <Stack gap="density-xxs" className="shrink-0">
        <Text kind="body/bold/md">Add a column</Text>
        <Text kind="body/regular/xs" className="text-secondary">
          Click or press Enter to add
        </Text>
      </Stack>

      <TextInput
        value={search}
        onValueChange={setSearch}
        placeholder="Search column types…"
        slotStart={<MagnifyingGlass size="14" className="text-secondary" />}
        className="shrink-0"
        attributes={{ Input: { 'aria-label': 'Search column types' } }}
      />

      <Stack gap="density-lg" className="min-h-0 flex-1 overflow-y-auto" paddingX="1">
        {filteredGroups.length === 0 ? (
          <Text kind="body/regular/sm" className="text-secondary">
            No column types match “{search}”.
          </Text>
        ) : (
          filteredGroups.map(({ group, options }) => (
            <ColumnTypeGroupSection
              key={group.id}
              group={group}
              options={options}
              onSelect={handleSelect}
              disabledReasons={disabledReasons}
            />
          ))
        )}
      </Stack>
    </Stack>
  );
};
