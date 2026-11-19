// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SliderWithTextInput } from '@nemo/common/src/components/SliderWithTextInput';
import type { HyperparameterFieldMetadata } from '@nemo/common/src/components/TrainingParameterSlider/types';
import { INFERENCE_HYPERPARAMETER_FIELD_METADATA } from '@nemo/common/src/constants/inferenceParameters';
import type { InferenceParams } from '@nemo/sdk/generated/platform/schema';
import { Stack } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

export const INFERENCE_PARAMETER_FIELDS = [
  'temperature',
  'max_tokens',
  'top_p',
] as const satisfies Array<keyof typeof INFERENCE_HYPERPARAMETER_FIELD_METADATA>;

export type InferenceParameterField = (typeof INFERENCE_PARAMETER_FIELDS)[number];

export type FieldMetadata = HyperparameterFieldMetadata<
  Record<InferenceParameterField, number>
>[InferenceParameterField];

export interface InferenceParametersProps {
  value: Partial<InferenceParams>;
  onChange: (value: Partial<InferenceParams>) => void;
  disabled?: boolean;
  /** Per-field overrides of the shared bounds and defaults, for callers whose backend differs. */
  fieldMetadata?: Partial<Record<InferenceParameterField, Partial<FieldMetadata>>>;
  /** The fields to show, for callers whose request only carries some of them. Defaults to all. */
  fields?: readonly InferenceParameterField[];
}

export const InferenceParameters: FC<InferenceParametersProps> = ({
  value,
  onChange,
  disabled,
  fieldMetadata,
  fields = INFERENCE_PARAMETER_FIELDS,
}) => (
  <Stack gap="4">
    {fields.map((key) => {
      const param = { ...INFERENCE_HYPERPARAMETER_FIELD_METADATA[key], ...fieldMetadata?.[key] };
      return (
        <SliderWithTextInput
          key={key}
          id={`${key}-slider`}
          field={{
            name: key,
            value: value[key] ?? param.default,
            onChange: (v: number | undefined) =>
              onChange({ ...value, [key]: v ?? value[key] ?? param.default }),
          }}
          defaultValue={param.default}
          disabled={disabled}
          min={param.min}
          max={param.max}
          step={param.step}
          size="compact"
          showReset
          formFieldProps={{ slotLabel: param.name, slotInfo: param.description }}
        />
      );
    })}
  </Stack>
);
