// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Trash } from '@nemo/icons/icons';
import { Badge, Button, Flex, Stack, Text, TextInput } from '@nvidia/foundations-react-core';
import { columnReference, type CustomTemplateRow } from '@studio/components/transform/template';
import { type FC } from 'react';

export interface CustomTemplateRowsProps {
  rows: readonly CustomTemplateRow[];
  columns: readonly string[];
  onChange: (rows: CustomTemplateRow[]) => void;
}

const isBlank = (row: CustomTemplateRow | undefined): boolean =>
  !row || (!row.key.trim() && !row.value.trim());

const isFilled = (row: CustomTemplateRow | undefined): boolean =>
  !!row && !!row.key.trim() && !!row.value.trim();

/** Grows the grid so an unfinished row is always available to type into. */
const withTrailingBlank = (rows: CustomTemplateRow[]): CustomTemplateRow[] => {
  const last = rows[rows.length - 1];
  return rows.length === 0 || isFilled(last) ? [...rows, { key: '', value: '' }] : rows;
};

/**
 * The escape hatch behind every preset: the raw `schema_transform` template as a
 * key/template grid. Keys accept dot paths (`inputs.instruction`) and numeric
 * segments (`messages.0.content`) to build nested objects and arrays. Filling in
 * the last row grows the grid, so no explicit add control is needed.
 */
export const CustomTemplateRows: FC<CustomTemplateRowsProps> = ({ rows, columns, onChange }) => {
  const update = (index: number, patch: Partial<CustomTemplateRow>) => {
    onChange(withTrailingBlank(rows.map((row, i) => (i === index ? { ...row, ...patch } : row))));
  };

  const remove = (index: number) => {
    onChange(withTrailingBlank(rows.filter((_, i) => i !== index)));
  };

  return (
    <Stack gap="density-md">
      {rows.map((row, index) => (
        <Flex key={index} align="center" gap="density-sm" className="min-w-0">
          <TextInput
            className="w-[220px] shrink-0 font-mono"
            value={row.key}
            placeholder="output key"
            aria-label={`Output key ${index + 1}`}
            onChange={(event) => update(index, { key: event.currentTarget.value })}
          />
          <TextInput
            className="min-w-0 flex-1 font-mono"
            value={row.value}
            placeholder="{{ column }} or any Jinja2 expression"
            aria-label={`Template for key ${index + 1}`}
            onChange={(event) => update(index, { value: event.currentTarget.value })}
          />
          <Button
            type="button"
            kind="tertiary"
            color="danger"
            size="small"
            aria-label={`Remove key ${index + 1}`}
            disabled={isBlank(row) && index === rows.length - 1}
            onClick={() => remove(index)}
          >
            <Trash size="14" aria-hidden />
          </Button>
        </Flex>
      ))}

      {columns.length > 0 && (
        <Flex align="center" gap="density-xs" className="min-w-0 flex-wrap">
          <Text kind="body/regular/xs" className="text-muted">
            Available columns:
          </Text>
          {columns.map((column) => (
            <Badge key={column} color="blue" kind="outline">
              {columnReference(column)}
            </Badge>
          ))}
        </Flex>
      )}
    </Stack>
  );
};
