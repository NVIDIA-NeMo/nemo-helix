// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useModelsGetModel } from '@nemo/sdk/generated/platform/models';
import { resolveAutomodelRecipe } from '@studio/util/forms/specDefaults';

/** Fetches the base model only while the recipe is `auto`; returns `auto` until it loads. */
export const useResolvedAutomodelRecipe = (
  modelRef: string | undefined,
  recipe: string | undefined,
  enabled: boolean
): string | undefined => {
  const parts = modelRef ? getPartsFromReference(modelRef) : null;
  const { data: model } = useModelsGetModel(parts?.workspace ?? '', parts?.name ?? '', undefined, {
    query: { enabled: enabled && recipe === 'auto' && Boolean(parts?.name), retry: false },
  });
  return resolveAutomodelRecipe(recipe, model?.spec);
};
