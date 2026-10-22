// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, FormField, Stack, Text, TextInput } from '@nvidia/foundations-react-core';
import { Trash, Close } from '@nvidia/react-gui-icons/icons';
import { ICON_COLOR_CLASS } from '@studio/components/AddColumnPalette/constants';
import { FieldControl } from '@studio/components/ColumnConfigPanel/FieldControl';
import { CardIconBadge } from '@studio/components/common/SelectableCard';
import {
  type BuilderColumn,
  getColumnFields,
  validateColumnName,
} from '@studio/routes/DataDesignerJobBuildRoute/columns';
import type { FC } from 'react';

export interface ColumnConfigPanelProps {
  column: BuilderColumn;
  takenNames: Set<string>;
  onChange: (patch: { name?: string; values?: Record<string, string> }) => void;
  onRemove: () => void;
  onClose: () => void;
}

/**
 * Right-hand config panel for the selected column on the build canvas.
 *
 * Renders as an inline pane (not an overlay) so the canvas stays visible while editing.
 * Edits are applied live via {@link ColumnConfigPanelProps.onChange} — as the user types
 * Jinja2 `{{ column_name }}` references, the canvas re-derives its dependency edges. The
 * fields shown are derived from the column type via {@link getColumnFields}.
 */
export const ColumnConfigPanel: FC<ColumnConfigPanelProps> = ({
  column,
  takenNames,
  onChange,
  onRemove,
  onClose,
}) => {
  const { option, name, values } = column;
  const { icon: Icon, label, description, color } = option;
  const fields = getColumnFields(option);
  const nameError = validateColumnName(name, takenNames);

  const setValue = (key: string, value: string) =>
    onChange({ values: { ...values, [key]: value } });

  return (
    <aside
      aria-label={`Configure ${label} column`}
      className="flex h-full w-full flex-col bg-surface-base"
    >
      <Flex
        align="start"
        justify="between"
        gap="density-md"
        className="shrink-0 border-b border-base p-density-lg"
      >
        <Flex align="center" gap="density-sm" className="min-w-0">
          <CardIconBadge>
            <Icon size="16" className={ICON_COLOR_CLASS[color]} aria-hidden />
          </CardIconBadge>
          <Stack gap="density-xxs" className="min-w-0">
            <Text kind="body/bold/md" className="truncate">
              {label}
            </Text>
            <Text kind="body/regular/xs" className="text-secondary truncate">
              {description}
            </Text>
          </Stack>
        </Flex>
        <Button
          kind="tertiary"
          color="neutral"
          size="small"
          aria-label="Close column config"
          onClick={onClose}
        >
          <Close size="16" aria-hidden />
        </Button>
      </Flex>

      <Stack gap="density-lg" padding="density-lg" className="min-h-0 flex-1 overflow-y-auto">
        <FormField
          slotLabel="Column name"
          required
          slotInfo="Other columns reference this via {{ name }}."
          status={name && nameError ? 'error' : undefined}
          slotError={name ? (nameError ?? undefined) : undefined}
        >
          <TextInput
            value={name}
            onValueChange={(value) => onChange({ name: value })}
            placeholder="e.g. topic"
            attributes={{ Input: { 'aria-label': 'Column name' } }}
          />
        </FormField>

        {fields.map((field) => (
          <FieldControl
            key={field.key}
            field={field}
            value={values[field.key] ?? ''}
            onChange={(value) => setValue(field.key, value)}
          />
        ))}
      </Stack>

      <Flex align="center" justify="start" className="shrink-0 border-t border-base p-density-lg">
        <Button kind="tertiary" color="danger" size="small" onClick={onRemove}>
          <Trash size="16" aria-hidden />
          Remove column
        </Button>
      </Flex>
    </aside>
  );
};
