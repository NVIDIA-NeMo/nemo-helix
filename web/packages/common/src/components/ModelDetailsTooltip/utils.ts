// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MODEL_METADATA, type ModelMetadata } from '@nemo/common/src/constants/modelMetadata';
import { getURNFromNamedEntityRef } from '@nemo/common/src/namedEntity';
import type { FinetuningType, ModelEntity } from '@nemo/sdk/generated/platform/schema';

interface TrainingOption {
  finetuning_type: FinetuningType;
  num_gpus: number;
}

interface ModelWithTrainingOptions {
  training_options?: TrainingOption[];
  namespace?: string;
  workspace?: string;
  name?: string;
  target?: {
    num_parameters?: number;
  };
}

export const parametersToString = (parameters: number, options?: { format?: 'short' | 'long' }) => {
  const format = options?.format ?? 'long';
  const isShort = format === 'short';
  if (parameters < 1_000) {
    return `${parameters}`;
  } else if (parameters < 1_000_000) {
    return `${(parameters / 1_000).toFixed()}${isShort ? 'K' : ' thousand'}`;
  } else if (parameters < 1_000_000_000) {
    return `${(parameters / 1_000_000).toFixed()}${isShort ? 'M' : ' million'}`;
  } else if (parameters < 1_000_000_000_000) {
    return `${(parameters / 1_000_000_000).toFixed()}${isShort ? 'B' : ' billion'}`;
  }
  return `${(parameters / 1_000_000_000_000).toFixed()}${isShort ? 'T' : ' trillion'}`;
};

export const getModelMetadata = (model: ModelEntity): ModelMetadata | undefined => {
  const modelId = getURNFromNamedEntityRef(model)?.split('@')[0];
  if (!modelId) {
    return undefined;
  }
  const metaBase = MODEL_METADATA[modelId];
  if ('target' in model && model.target && typeof model.target === 'object') {
    const modelWithOptions = model as unknown as ModelWithTrainingOptions;
    const trainingOptions = modelWithOptions.training_options || [];
    const recommendedGpus = trainingOptions.reduce(
      (acc: Record<FinetuningType, number>, option: TrainingOption) => {
        acc[option.finetuning_type] = option.num_gpus;
        return acc;
      },
      {} as Record<FinetuningType, number>
    );
    const fineTuneOptions: FinetuningType[] = Array.from(
      new Set(trainingOptions.map((option: TrainingOption) => option.finetuning_type))
    );
    const workspace = 'workspace' in model ? model.workspace : modelWithOptions.namespace;
    const numParameters = modelWithOptions.target?.num_parameters ?? 0;
    const additional = {
      name: model.name!,
      creator: workspace!,
      parameters: parametersToString(numParameters),
      'fine-tune-options': fineTuneOptions,
      'recommended-gpus-for-customization': recommendedGpus,
    };
    return { ...metaBase, ...additional };
  }
  return metaBase;
};
