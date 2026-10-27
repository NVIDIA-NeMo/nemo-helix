// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ResourceRef } from '@nemo/common/src/types';
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
 * A `workspace/name` model reference, as the evaluator API takes one for a target or judge.
 * The server resolves it to an address its jobs can reach; a URL built here would carry the
 * browser's origin instead. A bare name is qualified with `workspace`.
 *
 * @param workspace - The workspace a bare model name belongs to
 * @param modelRef - A model name, bare or already `workspace/name`
 */
export const toWorkspaceModelRef = (workspace: string, modelRef: string): string =>
  modelRef.includes('/') ? modelRef : `${workspace}/${modelRef}`;

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
