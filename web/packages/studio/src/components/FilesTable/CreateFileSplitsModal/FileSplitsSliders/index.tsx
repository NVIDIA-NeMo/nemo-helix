// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SliderWithTextInput } from '@nemo/common/src/components/SliderWithTextInput';
import { Stack, Button, Flex, Label } from '@nvidia/foundations-react-core';
import { LockOpen, LockClosed } from '@nvidia/react-gui-icons/icons';
import { FC, useCallback, useState } from 'react';
import { Controller, useFormContext, useWatch } from 'react-hook-form';

type SplitFormFields = {
  training: number;
  testing: number;
  validation: number;
};

/**
 * A component that allows the user to customize the split percentages for the file.
 * Handles automatic distribution of the splits as the user changes the sliders.
 * The three percentages (training, testing, validation) always sum to 100%.
 * Users can lock one slider at a time to fix its value and simplify redistribution.
 */
export const FileSplitsSliders: FC = () => {
  const { control, setValue } = useFormContext<SplitFormFields>();
  const watch = useWatch<SplitFormFields>();

  // Track which field is locked (only one at a time)
  const [lockedField, setLockedField] = useState<keyof SplitFormFields | null>(null);
  const [isFocused, setIsFocused] = useState<keyof SplitFormFields | null>(null);

  const toggleLock = useCallback((field: keyof SplitFormFields) => {
    setLockedField((prev) => {
      if (prev === field) {
        return null; // Unlock if already locked
      }
      return field; // LockClosed the new field (automatically unlocks previous)
    });
  }, []);

  const redistributePercentages = useCallback(
    (changedField: keyof SplitFormFields, newValue: number) => {
      // If no field is locked, distribute between the two unchanged fields
      if (lockedField === null) {
        const otherFields = ['training', 'testing', 'validation'].filter(
          (field) => field !== changedField
        ) as Array<keyof SplitFormFields>;

        const remainingPercentage = 100 - newValue;
        const equalShare = Math.round(remainingPercentage / 2);

        setValue(otherFields[0], equalShare, { shouldValidate: true });
        setValue(otherFields[1], remainingPercentage - equalShare, { shouldValidate: true });
        setValue(changedField, newValue, { shouldValidate: true });
        return;
      }

      // If the changed field is the locked field, adjust it to fit within constraints
      if (changedField === lockedField) {
        const otherFields = ['training', 'testing', 'validation'].filter(
          (field) => field !== changedField
        ) as Array<keyof SplitFormFields>;

        const otherValues = otherFields.map((field) => watch[field] ?? 0);
        const totalOther = otherValues.reduce((sum, val) => sum + val, 0);

        if (newValue + totalOther > 100) {
          setValue(changedField, 100 - totalOther, { shouldValidate: true });
        } else {
          setValue(changedField, newValue, { shouldValidate: true });
        }
        return;
      }

      // If a different field is locked, redistribute between the changed field and the other unlocked field
      const otherUnlockedField = ['training', 'testing', 'validation'].find(
        (field) => field !== changedField && field !== lockedField
      ) as keyof SplitFormFields;

      const lockedValue = watch[lockedField] ?? 0;
      const remainingPercentage = 100 - newValue - lockedValue;

      if (remainingPercentage < 0) {
        // Not enough percentage left, adjust the changed field
        setValue(changedField, 100 - lockedValue, { shouldValidate: true });
        setValue(otherUnlockedField, 0, { shouldValidate: true });
      } else {
        setValue(otherUnlockedField, remainingPercentage, { shouldValidate: true });
        setValue(changedField, newValue, { shouldValidate: true });
      }
    },
    [setValue, watch, lockedField]
  );

  const renderSliderWithLock = (
    fieldName: keyof SplitFormFields,
    label: string,
    isFocused: boolean
  ) => {
    const isLocked = lockedField === fieldName;

    return (
      <Controller
        control={control}
        name={fieldName}
        render={({ field }) => (
          <Flex gap="density-sm" align="center" className="*:flex-1">
            <SliderWithTextInput
              size="compact"
              field={field}
              defaultValue={field.value}
              min={0}
              max={100}
              step={1}
              disabled={isLocked}
              slotEnd={
                <Button
                  type="button"
                  kind="tertiary"
                  size="small"
                  onClick={() => toggleLock(fieldName)}
                  aria-label={isLocked ? `Unlock ${label}` : `LockClosed ${label}`}
                  className="p-2"
                >
                  {isLocked ? <LockClosed size="16" /> : <LockOpen size="16" />}
                </Button>
              }
              formFieldProps={{
                slotLabel: <Label className="leading-normal font-bold">{label}</Label>,
              }}
              attributes={{
                Slider: {
                  onValueChange: (value) => {
                    if (!isLocked) {
                      redistributePercentages(fieldName, value);
                    }
                  },
                },
                TextInput: {
                  name: fieldName,
                  value: isFocused ? field.value.toString() : field.value.toString() + '%',
                  type: 'text',
                  className: 'max-w-[64px]',
                  onFocus: () => setIsFocused(fieldName),
                  onBlur: () => setIsFocused(null),
                  onKeyDown: (e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                    }
                  },
                  onValueChange: (value) => {
                    if (!isLocked) {
                      const numValue = parseInt(value) || 0;
                      redistributePercentages(fieldName, numValue);
                    }
                  },
                },
              }}
            />
          </Flex>
        )}
      />
    );
  };

  return (
    <Stack gap="density-md">
      {renderSliderWithLock('training', 'Training %', isFocused === 'training')}
      {renderSliderWithLock('testing', 'Testing %', isFocused === 'testing')}
      {renderSliderWithLock('validation', 'Validation %', isFocused === 'validation')}
    </Stack>
  );
};
