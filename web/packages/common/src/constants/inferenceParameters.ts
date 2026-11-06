// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HyperparameterFieldMetadata } from '@nemo/common/src/components/TrainingParameterSlider/types';
import z from 'zod';

export const MAX_TOKENS_MIN = 1;
export const MAX_TOKENS_MAX = 4096;
export const MAX_TOKENS_STEP = 1;
export const MAX_TOKENS_DEFAULT = 1024;

export const MAX_COMPLETION_TOKENS_MIN = 1;
export const MAX_COMPLETION_TOKENS_MAX = 4096;
export const MAX_COMPLETION_TOKENS_STEP = 1;
export const MAX_COMPLETION_TOKENS_DEFAULT = 1024;

export const TEMPERATURE_MIN = 0.1;
export const TEMPERATURE_MAX = 2.0;
export const TEMPERATURE_STEP = 0.1;
export const TEMPERATURE_DEFAULT = 1.0;

export const TOP_P_MIN = 0.0;
export const TOP_P_MAX = 1.0;
export const TOP_P_STEP = 0.01;
export const TOP_P_DEFAULT = 1.0;

export interface InferenceSliderParams {
  temperature: number;
  max_tokens: number;
  max_completion_tokens: number;
  top_p: number;
}

// TODO (LLM-4626): Ideally, the default and min/max values are returned by the backend for each NIM,
// instead of hardcoded in the UI.
export const INFERENCE_HYPERPARAMETER_FIELD_METADATA: HyperparameterFieldMetadata<InferenceSliderParams> =
  {
    temperature: {
      name: 'Temperature',
      description:
        'Controls the creativity of the model. Higher values enable the model to generate more creative outputs, suitable for tasks such as creative writing. A value within the [0.5, 0.8] range is a good starting point for experimentation.',
      min: TEMPERATURE_MIN,
      max: TEMPERATURE_MAX,
      step: TEMPERATURE_STEP,
      default: TEMPERATURE_DEFAULT,
    },
    max_tokens: {
      name: 'Max Tokens',
      description:
        'The maximum number of tokens to generate. Tokens can be either an entire word, or parts of a word. For English, on average, 100 tokens form approximately 75 words.',
      min: MAX_TOKENS_MIN,
      max: MAX_TOKENS_MAX,
      step: MAX_TOKENS_STEP,
      default: MAX_TOKENS_DEFAULT,
    },
    max_completion_tokens: {
      name: 'Max Completion Tokens',
      description:
        'The maximum number of tokens to generate in the completion. An alternative to max_tokens with the same effect.',
      min: MAX_COMPLETION_TOKENS_MIN,
      max: MAX_COMPLETION_TOKENS_MAX,
      step: MAX_COMPLETION_TOKENS_STEP,
      default: MAX_COMPLETION_TOKENS_DEFAULT,
    },
    top_p: {
      name: 'Top P',
      description:
        'An alternative to sampling with temperature. The model considers tokens with top_p probability mass. Values closer to 1.0 allow more diverse outputs; values closer to 0 make outputs more focused.',
      min: TOP_P_MIN,
      max: TOP_P_MAX,
      step: TOP_P_STEP,
      default: TOP_P_DEFAULT,
    },
  };

export const modelInferenceParametersSchema = z.object({
  max_tokens: z
    .number()
    .min(MAX_TOKENS_MIN)
    .max(MAX_TOKENS_MAX)
    .step(MAX_TOKENS_STEP)
    .default(MAX_TOKENS_DEFAULT),
  temperature: z
    .number()
    .min(TEMPERATURE_MIN)
    .max(TEMPERATURE_MAX)
    .step(TEMPERATURE_STEP)
    .default(TEMPERATURE_DEFAULT),
  max_completion_tokens: z
    .number()
    .min(MAX_COMPLETION_TOKENS_MIN)
    .max(MAX_COMPLETION_TOKENS_MAX)
    .step(MAX_COMPLETION_TOKENS_STEP)
    .optional(),
  top_p: z.number().min(TOP_P_MIN).max(TOP_P_MAX).step(TOP_P_STEP).optional(),
});
