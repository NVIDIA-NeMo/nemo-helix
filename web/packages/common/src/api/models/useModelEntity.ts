// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useModelsGetModel } from '@nemo/sdk/generated/platform/models';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';

export interface UseModelEntityOptions {
  enabled?: boolean;
}

/**
 * Resolves a single model URN to its entity.
 *
 * Companion to `useModelSearch`: a paged dropdown only holds the models it has loaded, so a
 * selection restored from a URL or a form default has no entity attached. Callers that need
 * fields off the entity — `model_providers`, adapters, deployment state — fetch just that one
 * model instead of walking the catalogue to find it.
 *
 * Endpoint: GET /apis/models/v2/workspaces/{workspace}/models/{name}
 */
export const useModelEntity = (
  modelUrn: string | null | undefined,
  { enabled = true }: UseModelEntityOptions = {}
): ModelEntity | undefined => {
  const parts = modelUrn ? getPartsFromReference(modelUrn) : undefined;
  const { data } = useModelsGetModel(parts?.workspace ?? '', parts?.name ?? '', undefined, {
    query: { enabled: enabled && !!parts?.workspace && !!parts?.name },
  });
  return data;
};
