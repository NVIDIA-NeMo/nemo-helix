// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  InferenceParameterField,
  InferenceParametersProps,
} from '@nemo/common/src/components/ModelSelectV2/InferenceParameters';
import type { InferenceParams } from '@nemo/sdk/generated/platform/schema';

/** The chat runtime only puts temperature and max_tokens on the wire, so only those are offered. */
export const CHAT_INFERENCE_PARAM_FIELDS = [
  'temperature',
  'max_tokens',
] as const satisfies readonly InferenceParameterField[];

/** Chat's bounds differ from the shared ones: greedy decoding (0) is allowed, tokens move by 32. */
export const CHAT_INFERENCE_PARAM_METADATA = {
  temperature: { min: 0, max: 2, step: 0.05, default: 0.7 },
  max_tokens: { min: 32, max: 4096, step: 32, default: 512 },
} as const satisfies NonNullable<InferenceParametersProps['fieldMetadata']>;

export const DEFAULT_CHAT_INFERENCE_PARAMS: InferenceParams = {
  temperature: CHAT_INFERENCE_PARAM_METADATA.temperature.default,
  max_tokens: CHAT_INFERENCE_PARAM_METADATA.max_tokens.default,
};

/** Chat's bounds for a surface that starts at its own max_tokens, so reset returns there too. */
type ChatParamMetadata = Record<
  (typeof CHAT_INFERENCE_PARAM_FIELDS)[number],
  { min: number; max: number; step: number; default: number }
>;

export const chatParamMetadataWithMaxTokens = (maxTokens: number): ChatParamMetadata => ({
  ...CHAT_INFERENCE_PARAM_METADATA,
  max_tokens: { ...CHAT_INFERENCE_PARAM_METADATA.max_tokens, default: maxTokens },
});

/**
 * Apply params over the current ones, skipping any left unset (a null in a saved prompt): chat keeps
 * sending the last value rather than dropping the param or sending null.
 */
export const mergeChatParams = (
  previous: InferenceParams,
  next: Partial<InferenceParams>
): InferenceParams => ({
  ...previous,
  ...Object.fromEntries(Object.entries(next).filter(([, value]) => value != null)),
});
