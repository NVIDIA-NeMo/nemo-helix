// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Slider } from '@nvidia/foundations-react-core';
import type { ComponentProps } from 'react';

type SliderProps = ComponentProps<typeof Slider>;

/**
 * @deprecated - Prefer using a zod schema instead
 */
export interface CustomizerHyperparameters {
  batch_size: number;
  epochs: number;
  learning_rate: number;
  hidden_dropout: number;
  attention_dropout: number;
  ffn_dropout: number;
  weight_decay: number;
  // P-tuning only
  virtual_tokens: number;
  // Lora only
  adapter_dim: number;
  adapter_dropout: number;
}

export type HyperparameterFieldMetadata<T> = {
  readonly [K in keyof Required<T>]: {
    readonly name: string;
    readonly min: number;
    readonly max: number;
    readonly description?: string;
    readonly default: number;
    readonly step?: SliderProps['step'];
    readonly values?: readonly number[];
    readonly customSteps?: SliderProps['customSteps'];
  };
};

/**
 * Metadata about the different hyperparameters used by Customizer.
 */
export const HYPERPARAMETER_FIELD_METADATA: HyperparameterFieldMetadata<CustomizerHyperparameters> =
  {
    batch_size: {
      name: 'Batch Size',
      description:
        'Batch size is a hyperparameter used for training a customization. Batch size is the number of training samples used to train a single forward and backward pass.',
      min: 8,
      max: 128,
      default: 8,
      step: 8,
    },
    epochs: {
      name: 'Number of Epochs',
      description:
        'The number of times the entire dataset is propagated through the network during training.',
      min: 1,
      max: 100,
      default: 1,
      step: 1,
    },
    learning_rate: {
      name: 'Learning Rate',
      description: 'How much to adjust the model parameters in response to the loss gradient.',
      min: 1e-15,
      max: 1e-3,
      default: 1e-4,
      step: 1e-6,
    },
    hidden_dropout: {
      name: 'Hidden Dropout',
      description: 'Dropout probability for hidden state transformer.',
      min: 0.0,
      max: 1.0,
      default: 0.1,
      step: 0.01,
    },
    attention_dropout: {
      name: 'Attention Dropout',
      description: 'Dropout probability for attention.',
      min: 0.0,
      max: 1.0,
      default: 0.1,
      step: 0.01,
    },
    ffn_dropout: {
      name: 'FFN Dropout',
      description: 'Dropout probability in the feed-forward layer.',
      min: 0.0,
      max: 1.0,
      default: 0.1,
      step: 0.01,
    },
    weight_decay: {
      name: 'Weight Decay',
      description:
        'An additional penalty term added to the gradient descent to keep weights low and mitigate overfitting.',
      min: 0.0,
      max: 1.0,
      default: 0.01,
      step: 0.01,
    },
    virtual_tokens: {
      name: 'Number of Virtual Tokens',
      description:
        "Number of virtual tokens to use for customization. Virtual tokens are embeddings inserted into the model prompt that have no concrete mapping to strings or characters within the model's vocabulary.",
      min: 1,
      max: 100,
      default: 50,
      step: 1,
    },
    adapter_dim: {
      name: 'Adapter Dimensions',
      description:
        'Size of adapter layers added throughout the model. This is the size of the tunable layers that LoRA adds to various transformer blocks in the base model.',
      values: [8, 12, 16, 32, 64],
      min: 8,
      max: 64,
      default: 32,
    },
    adapter_dropout: {
      name: 'Adapter Dropout',
      description: 'Dropout probability in the adapter layer.',
      min: 0.0,
      max: 1.0,
      default: 0.1,
      step: 0.01,
    },
  };

/**
 * Validates the given value to ensure it equals one of the allowed values.
 *
 * @param fieldValue the training parameter's value
 * @param fieldMarks list of marks that represent the allowed values for the training parameter
 * @returns true if the field is valid; error message if not valid
 */
export const validateTrainingParameterIsAllowed = (fieldValue: number, fieldMarks: number[]) => {
  if (!Array.isArray(fieldMarks)) return true;

  const allowedValues = fieldMarks.map((mark) => mark);
  return (
    allowedValues.some((value) => value === fieldValue) ||
    `Invalid value. Please enter one of the following numbers: ${allowedValues.join(', ')}.`
  );
};

/**
 * Validates the given value to ensure it is within the range [min, max].
 *
 * @param fieldValue the training parameter's value
 * @param min minimum valid value, inclusive
 * @param max maximum valid value, inclusive
 * @returns true if the field is valid; error message if not valid
 */
export const validateTrainingParameterIsInRange = (
  fieldValue: number,
  min: number,
  max: number
) => {
  if (isNaN(fieldValue)) {
    return `Invalid value. Please enter a number between ${min} and ${max}.`;
  }
  return (
    (fieldValue >= min && fieldValue <= max) ||
    `Invalid value. Please enter a number between ${min} and ${max}.`
  );
};
