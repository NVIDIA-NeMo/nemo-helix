// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Accordion, FormField, Stack, TextArea } from '@nvidia/foundations-react-core';
import { DefaultSortControl } from '@studio/components/DefaultSortControl';
import { DEFAULT_SORT } from '@studio/components/DefaultSortControl/util';
import type { ExperimentSettingsFieldNames } from '@studio/components/evaluation/experimentSettings';
import { ExperimentFlagSwitch } from '@studio/components/ExperimentFlagSwitch';
import { Controller, type Control, type FieldValues } from 'react-hook-form';

interface ExperimentSettingsFieldsProps<T extends FieldValues> {
  control: Control<T>;
  names: ExperimentSettingsFieldNames<T>;
  disabled?: boolean;
  /** Evaluator names to offer as first-class default-sort fields. Only an existing experiment
   *  has any; a brand-new one has no evaluations to discover them from. */
  evaluatorOptions?: string[];
  /** Tuck the default sort and the two flags into a closed "Advanced Settings" section, for flows
   *  where most users take the defaults. */
  collapseAdvanced?: boolean;
}

/**
 * Description, default sort, and the two presentation flags — the whole of an Experiment below its
 * name. Rendered by the Experiments list's create modal, an experiment's edit modal, and the
 * evaluation form when it is creating an experiment, so all three offer the same settings.
 */
export const ExperimentSettingsFields = <T extends FieldValues>({
  control,
  names,
  disabled,
  evaluatorOptions,
  collapseAdvanced,
}: ExperimentSettingsFieldsProps<T>) => {
  const advanced = (
    <>
      <Controller
        control={control}
        name={names.defaultSort}
        render={({ field }) => (
          <DefaultSortControl
            value={(field.value as string | undefined) ?? DEFAULT_SORT}
            onChange={field.onChange}
            evaluatorOptions={evaluatorOptions}
            disabled={disabled}
          />
        )}
      />

      <Controller
        control={control}
        name={names.showEvaluationsOverTime}
        render={({ field }) => (
          <ExperimentFlagSwitch
            flag="show_evaluations_over_time"
            checked={Boolean(field.value)}
            onCheckedChange={field.onChange}
            onBlur={field.onBlur}
            disabled={disabled}
          />
        )}
      />

      <Controller
        control={control}
        name={names.isFavorite}
        render={({ field }) => (
          <ExperimentFlagSwitch
            flag="is_favorite"
            checked={Boolean(field.value)}
            onCheckedChange={field.onChange}
            onBlur={field.onBlur}
            disabled={disabled}
          />
        )}
      />
    </>
  );

  return (
    <Stack gap="density-2xl" className="w-full min-w-0">
      <Controller
        control={control}
        name={names.description}
        render={({ field, fieldState }) => (
          <FormField
            slotLabel="Description (Optional)"
            slotError={fieldState.error?.message}
            status={fieldState.error && 'error'}
          >
            <TextArea
              disabled={disabled}
              status={fieldState.error && 'error'}
              value={(field.value as string | undefined) ?? ''}
              onValueChange={(value: string) => field.onChange(value)}
              onBlur={field.onBlur}
            />
          </FormField>
        )}
      />

      {collapseAdvanced ? (
        <Accordion
          className="[&>div]:border-b-0"
          items={[
            {
              value: 'advanced',
              chevronPosition: 'start',
              slotTrigger: 'Advanced Settings',
              slotContent: (
                <Stack gap="density-2xl" className="pt-density-md">
                  {advanced}
                </Stack>
              ),
            },
          ]}
        />
      ) : (
        advanced
      )}
    </Stack>
  );
};
