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
import {
  isNemotronPersonasLocale,
  NEMOTRON_PERSONAS_LOCALES,
} from '@studio/components/agents/PersonaTestSet/nemotronPersonas';
import type { NemotronPersonasLocale } from '@studio/components/agents/PersonaTestSet/types';
import { useNemotronPersonasSetup } from '@studio/components/agents/PersonaTestSet/useNemotronPersonasSetup';
import { SeedDatasetConfig } from '@studio/components/ColumnConfigPanel/SeedDatasetConfig';
import { CardIconBadge } from '@studio/components/common/SelectableCard';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
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

const PersonasDatasetCheck: FC<{ locale: NemotronPersonasLocale }> = ({ locale }) => {
  const workspace = useWorkspaceFromPath();
  const { banner, secretModal } = useNemotronPersonasSetup(workspace, locale);
  return (
    <>
      {banner}
      {secretModal}
    </>
  );
};

/** Shown only when the person sampler can't read its locale's Nemotron Personas dataset. */
const PersonSamplerNote: FC<{ columnIndex: number }> = ({ columnIndex }) => {
  const { control } = useFormContext<JobBuilderFormValues>();
  // Data Designer samples en_US when no locale is set.
  const locale =
    useWatch({ control, name: `columns.${columnIndex}.values.locale` })?.trim() || 'en_US';

  return isNemotronPersonasLocale(locale) ? (
    <PersonasDatasetCheck key={locale} locale={locale} />
  ) : (
    <Banner kind="inline" status="warning" className="shrink-0">
      <Stack gap="density-xs">
        <Text kind="body/bold/sm">No Nemotron Personas dataset for {locale}</Text>
        <Text kind="body/regular/sm">
          Set <Text kind="body/bold/sm">Locale</Text> to one of{' '}
          {NEMOTRON_PERSONAS_LOCALES.join(', ')}.
        </Text>
      </Stack>
    </Banner>
  );
};

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
        {option.samplerType === SamplerType.person ? (
          <PersonSamplerNote columnIndex={columnIndex} />
        ) : null}
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
