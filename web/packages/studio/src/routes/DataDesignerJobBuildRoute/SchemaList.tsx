// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import {
  buildNameMaps,
  getColumnReferences,
  topologicalSortColumns,
} from '@studio/routes/DataDesignerJobBuildRoute/columns';
import { SchemaRow } from '@studio/routes/DataDesignerJobBuildRoute/SchemaRow';
import type { JobBuilderFormValues } from '@studio/routes/DataDesignerJobBuildRoute/useJobBuilder';
import { type FC, memo, useMemo } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

export interface SchemaListProps {
  selectedId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

/**
 * A flat, top-to-bottom list of the recipe's columns — a simpler alternative to the DAG
 * canvas. Each row shows the column's type and summary, with its dependencies listed
 * inline as relationship tags rather than drawn as edges. Selecting a row opens the same
 * config pane the canvas uses, so the surrounding left/right panels are unchanged.
 */
export const SchemaList: FC<SchemaListProps> = memo(function SchemaList({
  selectedId,
  onSelect,
  onDelete,
}) {
  const { control } = useFormContext<JobBuilderFormValues>();
  const columnRecord = useWatch({ control, name: 'columns' });
  const columns = useMemo(() => topologicalSortColumns(columnRecord), [columnRecord]);
  const referencesById = useMemo(() => {
    const { knownNames } = buildNameMaps(columns);
    return new Map(columns.map((column) => [column.id, getColumnReferences(column, knownNames)]));
  }, [columns]);

  return (
    <Stack className="h-full overflow-y-auto px-density-2xl py-density-xl">
      <Flex align="start" justify="between" gap="density-lg" className="mb-density-lg">
        <Stack gap="density-xxs" className="min-w-0">
          <Text kind="title/md" className="text-primary">
            Schema
          </Text>
          <Text kind="body/regular/sm" className="text-secondary">
            Reference columns with {'{{ column }}'}.
          </Text>
        </Stack>
      </Flex>

      {columns.length === 0 ? (
        <Flex
          align="center"
          justify="center"
          className="min-h-[160px] flex-1 rounded-md border border-dashed border-base"
        >
          <Text kind="body/regular/md" className="text-secondary">
            No columns yet — add one from the left to get started.
          </Text>
        </Flex>
      ) : (
        <Stack gap="density-md">
          {columns.map((column) => (
            <SchemaRow
              key={column.id}
              column={column}
              references={referencesById.get(column.id) ?? []}
              selected={column.id === selectedId}
              onSelect={onSelect}
              onDelete={onDelete}
            />
          ))}
        </Stack>
      )}
    </Stack>
  );
});
