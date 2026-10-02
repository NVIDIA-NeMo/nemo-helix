// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import {
  getModelsGetProviderQueryKey,
  modelsGetProvider,
} from '@nemo/sdk/generated/platform/model-providers';
import type {
  ModelEntity,
  ModelProvider,
  ServedModelMapping,
} from '@nemo/sdk/generated/platform/schema';
import { useQueries } from '@tanstack/react-query';
import { useMemo } from 'react';

export interface UseServedModelResult {
  /**
   * The provider's `served_models` entry for this id, if any provider serves it.
   * Its `served_model_name` is the backend's own name for the model — the value a
   * request must carry when it reaches the backend unrewritten (provider proxy).
   */
  servedModel?: ServedModelMapping;
  /** `<workspace>/<name>` ref of the provider that serves it. */
  providerRef?: string;
  /**
   * The provider that serves it. Callers need `model_deployment_id` to reach the
   * backing deployment; it is already fetched here, so hand it back rather than
   * making the caller re-resolve the ref.
   */
  provider?: ModelProvider;
  isLoading: boolean;
  /**
   * A provider referenced by `model.model_providers` could not be fetched. Only set
   * when no provider matched, where it is the difference between "nothing serves
   * this" and "we could not find out". Provider queries do not retry, so a single
   * 5xx, an authorization failure, or a stale provider reference lands here — and
   * reporting that as "not served" would be confidently wrong.
   */
  isError?: boolean;
}

/**
 * Look up how a model entity id is served, by walking `model.model_providers`.
 *
 * The mapping is discovered by the platform's provider reconciler rather than
 * derived, which matters for LoRA adapters: the backend's name for an adapter
 * (`{adapter_workspace}--{adapter_name}`, sometimes namespace-qualified, plus two
 * legacy shapes) is not reliably constructible on the client.
 *
 * @param model - Model entity whose providers to search
 * @param modelEntityId - Id to match against each provider's `served_models`
 */
export function useServedModel(
  model: ModelEntity | null | undefined,
  modelEntityId: string
): UseServedModelResult {
  const providerRefs = useMemo(() => model?.model_providers ?? [], [model]);
  const enabled = Boolean(model) && providerRefs.length > 0 && Boolean(modelEntityId);

  const queries = useQueries({
    queries: providerRefs.map((ref) => {
      const parts = getPartsFromReference(ref);
      return {
        queryKey: getModelsGetProviderQueryKey(parts.workspace, parts.name),
        queryFn: () => modelsGetProvider(parts.workspace, parts.name),
        enabled,
        retry: false,
        staleTime: 5 * 60 * 1000,
      };
    }),
  });

  return useMemo(() => {
    if (!enabled) {
      return { isLoading: false };
    }

    const isLoading = queries.some((q) => q.isLoading);

    for (const [index, query] of queries.entries()) {
      const provider: ModelProvider | undefined = query.data;
      const match = (provider?.served_models ?? []).find(
        (sm) => sm.model_entity_id === modelEntityId
      );
      if (match) {
        return { servedModel: match, providerRef: providerRefs[index], provider, isLoading };
      }
    }

    return { isLoading, isError: queries.some((q) => q.isError) };
  }, [enabled, queries, modelEntityId, providerRefs]);
}
