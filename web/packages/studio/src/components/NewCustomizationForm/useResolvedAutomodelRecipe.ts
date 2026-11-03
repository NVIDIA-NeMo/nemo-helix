// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useModelsGetModel } from '@nemo/sdk/generated/platform/models';
import { resolveAutomodelRecipe } from '@studio/util/forms/specDefaults';

/**
 * Fetches the base model only while the recipe is `auto`. Returns `null` until it loads,
 * as `auto` may still resolve to a retrieval recipe.
 */
export const useResolvedAutomodelRecipe = (
  modelRef: string | undefined,
  recipe: string | undefined,
  enabled: boolean
): string | null | undefined => {
  const parts = modelRef ? getPartsFromReference(modelRef) : null;
  const needsModel = enabled && recipe === 'auto';
  const { data: model } = useModelsGetModel(parts?.workspace ?? '', parts?.name ?? '', undefined, {
    query: { enabled: needsModel && Boolean(parts?.name), retry: false },
  });
  if (needsModel && !model) {
    return null;
  }
  return resolveAutomodelRecipe(recipe, model?.spec);
};
