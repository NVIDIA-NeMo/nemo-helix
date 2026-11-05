/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { MappingRow } from '@nemo/common/src/components/form/MappingFields/MappingRow';
import type {
  KeyValueComboboxPassthrough,
  KeyValueTextInputPassthrough,
} from '@nemo/common/src/components/form/MappingFields/types';
import { isDefined } from '@nemo/common/src/utils/isDefined';
import { Banner, Stack } from '@nvidia/foundations-react-core';
import { useEffect, useMemo } from 'react';
import {
  type Control,
  type FieldArrayPath,
  type FieldValues,
  useFieldArray,
  useFormState,
  useWatch,
} from 'react-hook-form';

const DEFAULT_SCHEMA_VALUE = (key: string) => `{{{${key}}}}`;

function isMappingRowEmpty(row: { key?: string; value?: string } | undefined): boolean {
  const k = typeof row?.key === 'string' ? row.key.trim() : '';
  const v = row?.value == null ? '' : String(row.value).trim();
  return !k && !v;
}

function getAtPath(obj: unknown, path: string): unknown {
  if (!obj || typeof obj !== 'object') return undefined;
  const parts = path.split('.');
  let cur: unknown = obj;
  for (const p of parts) {
    if (cur === null || cur === undefined || typeof cur !== 'object') return undefined;
    cur = (cur as Record<string, unknown>)[p];
  }
  return cur;
}

function fieldArrayHasErrors(arrayErrors: unknown): boolean {
  return Array.isArray(arrayErrors) && arrayErrors.some(isDefined);
}

function firstKeyValueRowMessage(arrayErrors: unknown): string | undefined {
  if (!Array.isArray(arrayErrors)) return undefined;
  for (const item of arrayErrors) {
    if (!item || typeof item !== 'object') continue;
    const row = item as Record<string, { message?: string } | undefined>;
    const keyMsg = row.key?.message;
    if (keyMsg) return keyMsg;
    const valueMsg = row.value?.message;
    if (valueMsg) return valueMsg;
  }
  return undefined;
}

/** Stable identity so memoized rows are not invalidated when no overrides are passed. */
const NO_OVERRIDES = {};

export interface MappingFieldsProps<
  TFieldValues extends FieldValues,
  TName extends FieldArrayPath<TFieldValues>,
> {
  control: Control<TFieldValues>;
  /**
   * react-hook-form field array path; each item is `{ key: string; value?: string }`.
   * The UI keeps one trailing blank row; entering text in key or value on that row appends another blank row.
   * Consumers should omit all-blank rows from API payloads (see Transform file submit).
   */
  name: TName;
  /**
   * When true, disables mapping inputs and row removal. Also disabled when the form is disabled
   * (`useForm({ disabled: true })` / `FormProvider`).
   */
  disabled?: boolean;
  /**
   * When set (e.g. file JSON schema), replaces the field array whenever the sorted list of keys changes.
   * Row values are `schemaValueForKey(key)` (default: `{{{key}}}`), matching the file-transform mapping UX.
   * Record values are not used; only keys matter.
   */
  schema?: Record<string, unknown>;
  /** Default value string for each row when syncing from `schema`. Keep stable (e.g. module-level fn) if customized. */
  schemaValueForKey?: (key: string) => string;
  /**
   * Combobox suggestion lists; default key/value options are derived from `schema` keys when present.
   * When a column has no suggestions (empty array and no schema-derived items), that column uses a plain text input instead of a combobox.
   */
  keySuggestions?: string[];
  valueSuggestions?: string[];
  keyColumnLabel?: string;
  valueColumnLabel?: string;
  /** Forward props to the key/value field controls (combobox vs text input is chosen automatically). */
  attributes?: {
    keyCombobox?: Partial<KeyValueComboboxPassthrough>;
    valueCombobox?: Partial<KeyValueComboboxPassthrough>;
    keyTextInput?: Partial<KeyValueTextInputPassthrough>;
    valueTextInput?: Partial<KeyValueTextInputPassthrough>;
  };
}

export const MappingFields = <
  TFieldValues extends FieldValues,
  TName extends FieldArrayPath<TFieldValues>,
>({
  control,
  name,
  disabled,
  schema,
  schemaValueForKey = DEFAULT_SCHEMA_VALUE,
  keySuggestions: keySuggestionsProp,
  valueSuggestions: valueSuggestionsProp,
  keyColumnLabel = 'Key',
  valueColumnLabel = 'Value',
  attributes,
}: MappingFieldsProps<TFieldValues, TName>) => {
  const nameStr = name as string;
  const { errors, disabled: formDisabled } = useFormState<TFieldValues>({ control });
  const isDisabled = Boolean(disabled) || formDisabled;
  const arrayErrors = getAtPath(errors, nameStr);

  const {
    fields: rows,
    append,
    remove,
    replace,
  } = useFieldArray({
    control,
    name,
  });

  const watchedRows = useWatch({
    control,
    name: name as never,
  }) as Array<{ key?: string; value?: string }> | undefined;

  useEffect(() => {
    if (isDisabled) return;
    const list = Array.isArray(watchedRows) ? watchedRows : [];
    if (list.length === 0) {
      append({ key: '', value: '' } as Parameters<typeof append>[0]);
      return;
    }
    if (list.length >= 2) {
      const secondLast = list[list.length - 2];
      const last = list[list.length - 1];
      if (isMappingRowEmpty(secondLast) && isMappingRowEmpty(last)) {
        remove(list.length - 2);
        return;
      }
    }
    const lastRow = list[list.length - 1];
    if (!isMappingRowEmpty(lastRow)) {
      append({ key: '', value: '' } as Parameters<typeof append>[0]);
    }
  }, [append, isDisabled, remove, watchedRows]);

  const schemaKeySignature = useMemo(
    () => (schema ? JSON.stringify(Object.keys(schema).sort()) : ''),
    [schema]
  );

  useEffect(() => {
    if (schema === undefined) return;
    replace([
      ...Object.keys(schema).map((key) => ({
        key,
        value: schemaValueForKey(key),
      })),
      { key: '', value: '' },
    ] as Parameters<typeof replace>[0]);
  }, [replace, schema, schemaKeySignature, schemaValueForKey]);

  const keyOpts = useMemo(
    () => keySuggestionsProp ?? (schema ? Object.keys(schema) : []),
    [keySuggestionsProp, schema]
  );

  const valueOpts = useMemo(
    () =>
      valueSuggestionsProp ??
      (schema ? Object.keys(schema).map((key) => schemaValueForKey(key)) : []),
    [valueSuggestionsProp, schema, schemaValueForKey]
  );

  const keyComboboxProps = (attributes?.keyCombobox ??
    NO_OVERRIDES) as Partial<KeyValueComboboxPassthrough>;
  const valueComboboxProps = (attributes?.valueCombobox ??
    NO_OVERRIDES) as Partial<KeyValueComboboxPassthrough>;
  const keyTextInputProps = (attributes?.keyTextInput ??
    NO_OVERRIDES) as Partial<KeyValueTextInputPassthrough>;
  const valueTextInputProps = (attributes?.valueTextInput ??
    NO_OVERRIDES) as Partial<KeyValueTextInputPassthrough>;

  const firstFieldError = firstKeyValueRowMessage(arrayErrors);

  return (
    <Stack gap="density-lg">
      <Stack gap="density-lg">
        {rows.map((row, index) => (
          <MappingRow
            key={row.id}
            control={control}
            name={nameStr}
            index={index}
            isLastRow={index === rows.length - 1}
            isDisabled={isDisabled}
            keyOpts={keyOpts}
            valueOpts={valueOpts}
            keyColumnLabel={keyColumnLabel}
            valueColumnLabel={valueColumnLabel}
            keyCombobox={keyComboboxProps}
            valueCombobox={valueComboboxProps}
            keyTextInput={keyTextInputProps}
            valueTextInput={valueTextInputProps}
            onRemove={remove}
          />
        ))}
      </Stack>
      {fieldArrayHasErrors(arrayErrors) ? (
        <Banner
          kind="inline"
          status="warning"
          attributes={{ BannerIcon: { className: 'self-start' } }}
        >
          {firstFieldError}
        </Banner>
      ) : null}
    </Stack>
  );
};
