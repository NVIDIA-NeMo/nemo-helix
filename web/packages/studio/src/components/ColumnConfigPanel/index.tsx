// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledSwitch } from '@nemo/common/src/components/form/ControlledSwitch';
import { ControlledTextArea } from '@nemo/common/src/components/form/ControlledTextArea';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { Trash, Close } from '@nemo/icons/icons';
import { SamplerType } from '@nemo/sdk/generated/data-designer/schema';
import { Banner, Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { ICON_COLOR_CLASS } from '@studio/components/AddColumnPalette/constants';
import { SeedDatasetConfig } from '@studio/components/ColumnConfigPanel/SeedDatasetConfig';
import { CardIconBadge } from '@studio/components/common/SelectableCard';
import {
  type ColumnField,
  getColumnFields,
  validateColumnName,
} from '@studio/routes/DataDesignerJobBuildRoute/columns';
import type { JobBuilderFormValues } from '@studio/routes/DataDesignerJobBuildRoute/useJobBuilder';
import type { FC } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

interface FieldControlProps {
  columnIndex: number;
  field: ColumnField;
}

/** A field-level RHF subscription; editing it leaves the route, list, and other fields alone. */
const FieldControl: FC<FieldControlProps> = ({ columnIndex, field }) => {
  const fieldPath = `columns.${columnIndex}.values.${field.key}` as const;
  const useControllerProps = { name: fieldPath };
  const formFieldProps = { slotInfo: field.helperText };

  switch (field.kind) {
    case 'textarea':
      return (
        <ControlledTextArea
          label={field.label}
          required={field.required}
          useControllerProps={useControllerProps}
          formFieldProps={formFieldProps}
          placeholder={field.placeholder}
          resizeable="auto"
        />
      );
    case 'select':
      return (
        <ControlledSelect
          items={
            field.options?.map((option) => ({ children: option.label, value: option.value })) ?? []
          }
          useControllerProps={useControllerProps}
          formFieldProps={{ ...formFieldProps, slotLabel: field.label, required: field.required }}
          placeholder="Select…"
        />
      );
    case 'switch':
      return (
        <ControlledSwitch
          useControllerProps={useControllerProps}
          formFieldProps={{ ...formFieldProps, slotLabel: field.label }}
          toChecked={(value) => value === 'true'}
          toFormValue={(checked) => (checked ? 'true' : 'false')}
        />
      );
    default:
      return (
        <ControlledTextInput
          label={field.label}
          required={field.required}
          useControllerProps={useControllerProps}
          formFieldProps={formFieldProps}
          placeholder={field.placeholder}
          type={field.kind === 'number' ? 'text' : undefined}
          attributes={field.kind === 'number' ? { Input: { inputMode: 'decimal' } } : undefined}
        />
      );
  }
};

interface ColumnNameControlProps {
  columnIndex: number;
}

const ColumnNameControl: FC<ColumnNameControlProps> = ({ columnIndex }) => {
  const { control, getValues } = useFormContext<JobBuilderFormValues>();
  const namePath = `columns.${columnIndex}.name` as const;
  const columnCount = getValues('columns').length;
  const value = useWatch({ control, name: namePath }) ?? '';
  const names = useWatch({
    control,
    name: Array.from({ length: columnCount }, (_, index) => `columns.${index}.name` as const),
  });
  const takenNames = new Set(names.filter((name, index) => index !== columnIndex && Boolean(name)));
  const nameError = validateColumnName(value, takenNames);

  return (
    <ControlledTextInput
      label="Column name"
      required
      useControllerProps={{ name: namePath }}
      formFieldProps={{
        slotInfo: 'Other columns reference this via {{ name }}.',
        status: value && nameError ? 'error' : undefined,
        slotError: value ? (nameError ?? undefined) : undefined,
      }}
      placeholder="e.g. topic"
      aria-label="Column name"
    />
  );
};

export interface ColumnConfigPanelProps {
  columnId: string;
  onRemove: () => void;
  onClose: () => void;
}

const PersonSamplerNote: FC = () => (
  <Banner kind="inline" status="info" title="Requires a managed dataset">
    <Text kind="body/regular/sm">
      The person sampler reads from downloaded Nemotron Personas datasets. Before it can preview or
      build:
    </Text>
    <ol className="list-decimal pl-density-lg">
      <li>
        Download the Nemotron Personas dataset for your locale into the managed assets directory.
      </li>
      <li>
        Set <Text kind="body/bold/sm">Locale</Text> below to a supported value (e.g. en_US, en_IN,
        fr_FR, ja_JP, ko_KR, pt_BR).
      </li>
    </ol>
  </Banner>
);

/** Right-hand config panel for one selected column. */
export const ColumnConfigPanel: FC<ColumnConfigPanelProps> = ({ columnId, onRemove, onClose }) => {
  const { getValues } = useFormContext<JobBuilderFormValues>();
  const columnIndex = getValues('columns').findIndex((column) => column.id === columnId);
  const column = getValues(`columns.${columnIndex}`);
  const { option } = column;
  const { icon: Icon, label, description, color } = option;
  const fields = getColumnFields(option);

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
        {option.samplerType === SamplerType.person ? <PersonSamplerNote /> : null}
        <ColumnNameControl columnIndex={columnIndex} />
        {option.columnType === 'seed-dataset' ? (
          <SeedDatasetConfig columnIndex={columnIndex} />
        ) : (
          fields.map((field) => (
            <FieldControl key={field.key} columnIndex={columnIndex} field={field} />
          ))
        )}
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
