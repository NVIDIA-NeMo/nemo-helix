// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Button,
  Flex,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
  Stack,
  Tag,
  Text,
  TextInput,
} from '@nvidia/foundations-react-core';
import { Code } from '@nvidia/react-gui-icons/icons';
import type { OutputFormatField } from '@studio/components/transform/formats';
import { columnReference } from '@studio/components/transform/template';
import { type FC } from 'react';

export interface FieldMappingRowProps {
  field: OutputFormatField;
  /** The Jinja2 template currently mapped to this field, `''` when unmapped. */
  value: string;
  /** Source column names available in the file being transformed. */
  columns: readonly string[];
  /** When true the row shows a raw template input instead of the column picker. */
  isRaw: boolean;
  /**
   * Name of the UUID column the job would generate. Offered in the picker for
   * identity fields, so a source with no unique key still gets a per-row id.
   */
  generatedIdColumn?: string;
  onChange: (path: string, value: string) => void;
  onToggleRaw: (path: string) => void;
}

/**
 * One field of an output format. The common case — "this field is that column" —
 * is a column picker; the `{ }` toggle drops to a raw Jinja2 input for anything
 * the picker cannot express (filters, concatenation, literals).
 */
export const FieldMappingRow: FC<FieldMappingRowProps> = ({
  field,
  value,
  columns,
  isRaw,
  generatedIdColumn,
  onChange,
  onToggleRaw,
}) => {
  const offersGeneratedId = Boolean(field.identity && generatedIdColumn);
  const options = offersGeneratedId ? [...columns, generatedIdColumn as string] : columns;
  const selectedColumn = options.find((column) => columnReference(column) === value) ?? '';
  const isGenerated = offersGeneratedId && selectedColumn === generatedIdColumn;
  const isUnmappedRequired = field.required && !value.trim();

  return (
    <Stack gap="density-xs" className="min-w-0">
      <Flex align="center" gap="density-sm" className="min-w-0">
        <Text kind="body/semibold/sm" className="truncate font-mono">
          {field.label}
        </Text>
        <Tag color={field.required ? 'red' : 'gray'} kind="outline" density="compact" readOnly>
          {field.required ? 'Required' : 'Optional'}
        </Tag>
        {isGenerated && (
          <Tag color="teal" kind="outline" density="compact" readOnly>
            Generated
          </Tag>
        )}
      </Flex>

      <Flex align="center" gap="density-sm" className="min-w-0">
        {isRaw ? (
          <TextInput
            className="min-w-0 flex-1 font-mono"
            value={value}
            placeholder="{{ column }} or any Jinja2 expression"
            aria-label={`${field.label} template`}
            onChange={(event) => onChange(field.path, event.currentTarget.value)}
          />
        ) : (
          <SelectRoot
            value={selectedColumn}
            onValueChange={(column: string) =>
              onChange(field.path, column ? columnReference(column) : '')
            }
          >
            <SelectTrigger
              className="min-w-0 flex-1"
              placeholder="Choose a source column"
              aria-label={`${field.label} source column`}
            />
            <SelectContent className="w-(--radix-popper-anchor-width)">
              <SelectListbox>
                {columns.map((column) => (
                  <SelectItem key={column} value={column}>
                    {column}
                  </SelectItem>
                ))}
                {offersGeneratedId && (
                  <SelectItem value={generatedIdColumn as string}>
                    {`${generatedIdColumn} — generate a UUID per row`}
                  </SelectItem>
                )}
              </SelectListbox>
            </SelectContent>
          </SelectRoot>
        )}
        <Button
          type="button"
          kind={isRaw ? 'secondary' : 'tertiary'}
          color="neutral"
          size="small"
          aria-label={`${isRaw ? 'Pick a column for' : 'Write a template for'} ${field.label}`}
          onClick={() => onToggleRaw(field.path)}
        >
          <Code size="14" aria-hidden />
        </Button>
      </Flex>

      <Text
        kind="body/regular/xs"
        className={isUnmappedRequired ? 'text-accent-yellow' : 'text-muted'}
      >
        {isUnmappedRequired
          ? `${field.description} This field is required and has no source.`
          : isGenerated
            ? `${field.description} The source has no unique key, so the job adds a ${generatedIdColumn} column.`
            : field.description}
      </Text>
    </Stack>
  );
};
