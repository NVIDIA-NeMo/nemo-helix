// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ResourceRef } from '@nemo/common/src/types';
import { getGatewayProxyGetQueryKey } from '@nemo/sdk/generated/platform/inference-gateway';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ChatCompletionTool } from 'openai/resources/index.mjs';

/**
 * Get the tools for a model
 *
 * @param model - The model to get the tools for
 * @returns The tools for the model
 */
export const getModelTools = (model: ModelEntity): ChatCompletionTool[] => {
  return model.custom_fields?.tools && typeof model.custom_fields.tools === 'string'
    ? JSON.parse(model.custom_fields.tools)
    : [];
};

/**
 * Build an inference gateway URL for a model entity.
 * Uses the SDK's gateway proxy path to route through the inference gateway.
 *
 * @param workspace - The workspace the model belongs to
 * @param modelName - The model entity name (without workspace prefix)
 * @param isChat - Whether to use chat completions (default true)
 */
export const getModelInferenceGatewayUrl = (
  workspace: string,
  modelRef: string,
  isChat: boolean = true
): string => {
  // Strip workspace prefix if model ref is in "workspace/modelName" format
  const modelName = modelRef.includes('/') ? modelRef.split('/').slice(1).join('/') : modelRef;
  const trailingUri = isChat ? 'v1/chat/completions' : 'v1/completions';
  const [path] = getGatewayProxyGetQueryKey(workspace, modelName, trailingUri);
  return `${PLATFORM_BASE_URL}${path}`;
};

/**
 * Build a model config for evaluation related types like targets and metrics
 * @param model_id - The model URN to build the config for
 * @returns The model config
 */
export const buildModelConfig = (model_id: ResourceRef, isChat: boolean = true) => {
  return {
    api_endpoint: {
      url: isChat
        ? `${PLATFORM_BASE_URL}/v1/chat/completions`
        : `${PLATFORM_BASE_URL}/v1/completions`,
      model_id,
      format: 'nim' as const,
    },
  };
};
