// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { toInferenceModelEntityId } from '@nemo/common/src/utils/models';
import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { useServedModel } from '@studio/hooks/useServedModel';

interface UseModelIsServedResult {
  /** Whether at least one provider lists this model in its served_models. */
  isServed: boolean;
  isLoading: boolean;
  /**
   * A provider could not be fetched, so `isServed: false` means "we could not find
   * out", not "nothing serves this". Callers that state the negative affirmatively
   * must exclude this case.
   */
  isError: boolean;
}

/**
 * Whether a provider actually serves this model — or, with `adapter`, that adapter
 * on this model.
 *
 * The two are distinct states: a deployment can be READY and serving the base model
 * while the adapter it was asked to load is not (yet) among its `served_models`.
 * Asking about the base in that case answers the wrong question.
 *
 * @param model - The model entity, or the adapter's base model
 * @param adapter - The adapter to ask about, if the question is about one
 */
export function useModelIsServed(
  model: ModelEntity | null | undefined,
  adapter?: Adapter | null
): UseModelIsServedResult {
  const modelEntityId = model ? toInferenceModelEntityId(model, adapter) : '';
  const { servedModel, isLoading, isError } = useServedModel(model, modelEntityId);

  return { isServed: Boolean(servedModel), isLoading, isError: Boolean(isError) };
}
