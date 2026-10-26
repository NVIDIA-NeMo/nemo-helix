// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { toScientificNotation } from '@nemo/common/src/utils/formatters';
import { clamp } from '@nemo/common/src/utils/math';
import { InfoCircle, Retry } from '@nemo/icons/icons';
import {
  Block,
  Button,
  Flex,
  FormField,
  FormFieldProps,
  Slider,
  Stack,
  Text,
  TextInput,
  TextInputProps,
  Tooltip,
} from '@nvidia/foundations-react-core';
import { ComponentProps, ReactNode, useState } from 'react';
import { FieldValues } from 'react-hook-form';

type SliderProps = ComponentProps<typeof Slider>;

export type SliderWithTextInputProps = {
  field: FieldValues;
  defaultValue?: number;
  unsetPlaceholder?: string;
  max: number;
  min: number;
  step?: number;
  id?: string;
  disabled?: boolean;
  showStepMarkers?: boolean;
  showReset?: boolean;
  size?: 'normal' | 'compact';
  attributes?: {
    Slider?: Partial<SliderProps>;
    TextInput?: Partial<TextInputProps>;
  };
  formFieldProps?: FormFieldProps;
  slotEnd?: ReactNode;
  displayName?: string;
};

export const SliderWithTextInput = ({
  field,
  defaultValue,
  unsetPlaceholder,
  formFieldProps,
  max,
  min,
  step,
  id,
  disabled,
  showStepMarkers = false,
  showReset = true,
  size = 'normal',
  attributes,
  slotEnd,
  displayName,
}: SliderWithTextInputProps) => {
  const [draftValue, setDraftValue] = useState<string | null>(null);

  const handleSliderChange = (newValue: number) => {
    const clampedValue = clamp(newValue, min, max);
    setDraftValue(null);
    field.onChange(clampedValue);
    attributes?.Slider?.onValueChange?.(clampedValue);
  };
  const handleTextInputChange = (newValue: string, event: React.ChangeEvent<HTMLInputElement>) => {
    setDraftValue(newValue);
    if (event.target.validity?.badInput) return;
    if (newValue === '') {
      field.onChange(undefined);
      attributes?.TextInput?.onValueChange?.('', event);
      return;
    }
    const numberValue = parseFloat(newValue);
    if (Number.isNaN(numberValue)) return;
    if (numberValue < min || numberValue > max) return;
    field.onChange(numberValue);
    attributes?.TextInput?.onValueChange?.(numberValue.toString(), event);
  };
  const handleTextInputBlur = (event: React.FocusEvent<HTMLInputElement>) => {
    const rawValue = draftValue;
    setDraftValue(null);
    attributes?.TextInput?.onBlur?.(event);
    if (rawValue === null || rawValue === '') return;
    const numberValue = parseFloat(rawValue);
    if (Number.isNaN(numberValue)) return;
    const clampedValue = clamp(numberValue, min, max);
    if (clampedValue === field.value) return;
    field.onChange(clampedValue);
    attributes?.Slider?.onValueChange?.(clampedValue);
  };
  const handleReset = () => {
    setDraftValue(null);
    field.onChange(defaultValue);
    // Nothing numeric to forward when the field resets to unset.
    if (defaultValue !== undefined) attributes?.Slider?.onValueChange?.(defaultValue);
  };
  const fallback = defaultValue ?? min;
  const isFieldValueNumber = typeof field.value === 'number' && !Number.isNaN(field.value);
  const safeFieldValue = isFieldValueNumber ? field.value : fallback;
  const textInputValue = draftValue ?? (isFieldValueNumber ? field.value.toString() : '');

  const stepMarkerClassNames =
    'pb-5 [&_.nv-slider-step:first-of-type]:items-start [&_.nv-slider-step:last-of-type]:items-end';

  // Size-specific dimensions
  const fieldStatus = formFieldProps?.slotError ? ('error' as const) : undefined;
  const labelWidth = 'w-[165px]';
  const sliderMinWidth = size === 'compact' ? 'min-w-[100px]' : 'min-w-[200px]';
  const textInputWidth = size === 'compact' ? 'w-[80px]' : 'w-[120px]';
  const gap = size === 'compact' ? 'density-sm' : 'density-md';

  const labelNode = formFieldProps?.slotLabel && (
    <Flex direction="row" align="center" gap="density-xs" className="shrink-0">
      <Text
        kind={size === 'compact' ? 'label/bold/sm' : 'label/regular/md'}
        className={size === 'compact' ? undefined : 'truncate'}
      >
        {formFieldProps.slotLabel}
      </Text>
      {formFieldProps.slotInfo && (
        <Tooltip slotContent={formFieldProps.slotInfo} side="top">
          <InfoCircle size="12" className="shrink-0" />
        </Tooltip>
      )}
    </Flex>
  );

  const sliderRow = (
    <Flex align="center" gap={gap} className="w-full">
      {size === 'normal' && labelNode && (
        <Flex direction="row" align="center" gap="density-xs" className={`${labelWidth} shrink-0`}>
          <Text kind="label/regular/md" className="truncate">
            {formFieldProps?.slotLabel}
          </Text>
          {formFieldProps?.slotInfo && (
            <Tooltip slotContent={formFieldProps.slotInfo} side="top">
              <InfoCircle size="12" className="shrink-0" />
            </Tooltip>
          )}
        </Flex>
      )}

      <Block
        className={`flex-1 ${sliderMinWidth} ${(showStepMarkers && stepMarkerClassNames) || ''}`}
      >
        <Slider
          orientation="horizontal"
          value={safeFieldValue}
          max={max}
          min={min}
          step={step}
          disabled={disabled}
          stepPosition={showStepMarkers ? (attributes?.Slider?.stepPosition ?? 'end') : undefined}
          customSteps={attributes?.Slider?.customSteps ?? [min, max]}
          aria-label="Controlled slider"
          stepFormatFn={attributes?.Slider?.stepFormatFn ?? toScientificNotation}
          {...attributes?.Slider}
          onValueChange={handleSliderChange}
        />
      </Block>

      <TextInput
        name={field.name}
        status={fieldStatus}
        aria-label="Slider value"
        value={textInputValue}
        placeholder={unsetPlaceholder}
        max={max.toString()}
        min={min.toString()}
        step={step?.toString()}
        type="number"
        disabled={disabled}
        className={`${textInputWidth} h-[40px] shrink-0`}
        {...attributes?.TextInput}
        onValueChange={handleTextInputChange}
        onBlur={handleTextInputBlur}
        attributes={{
          Input: {
            'aria-label': `${id || 'slider'}_text_input`,
            className: 'text-center',
          },
        }}
      />

      {showReset && (
        <Button
          kind="tertiary"
          size="small"
          aria-label={`Reset ${displayName || field.name} to default value`}
          disabled={disabled}
          onClick={handleReset}
          className="shrink-0"
          type="button"
        >
          <Retry />
        </Button>
      )}
      {slotEnd}
    </Flex>
  );

  return (
    <FormField
      name={field.name}
      status={formFieldProps?.slotError ? 'error' : undefined}
      {...formFieldProps}
      slotLabel={undefined}
      slotInfo={undefined}
    >
      {() => {
        if (size === 'compact') {
          return (
            <Stack gap="density-xs" className="w-full">
              {labelNode}
              {sliderRow}
            </Stack>
          );
        }
        return sliderRow;
      }}
    </FormField>
  );
};
