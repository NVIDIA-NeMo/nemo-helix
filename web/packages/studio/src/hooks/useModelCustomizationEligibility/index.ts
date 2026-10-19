// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';

/**
 * Whether a model can be fine-tuned: it needs a fileset holding the base weights.
 * All three customization backends require one at job-compile time and reject the
 * job with a 422 otherwise, so this is also what the base-model pickers filter on.
 */
export const canFineTuneModel = (model: ModelEntity | null | undefined): boolean =>
  Boolean(model?.fileset);

export interface ModelCustomizationEligibility {
  canFineTune: boolean;
  canCustomize: boolean;
  isLoading: boolean;
}

/**
 * Whether the given model can be fine-tuned.
 *
 * Fine-tuning requires `model.fileset` to be populated (the fileset holds the
 * weights/config needed as the training starting point).
 */
export const useModelCustomizationEligibility = (
  model: ModelEntity | undefined
): ModelCustomizationEligibility => {
  const canFineTune = canFineTuneModel(model);

  return {
    canFineTune,
    canCustomize: canFineTune,
    isLoading: false,
  };
};
