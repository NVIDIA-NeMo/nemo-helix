// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  ModelEntity,
  ModelEntitysPage,
  ModelsListModelsParams,
} from '@nemo/sdk/generated/platform/schema';

export const entityStoreBaseModel1: ModelEntity = {
  id: 'model-base-1',
  created_at: '2025-01-07T16:08:49.525322',
  updated_at: '2025-01-07T16:08:49.525324',
  name: 'codellama-70b',
  workspace: 'meta',
  description: 'None',
  spec: {
    checkpoint_model_name: 'model',
    family: 'llama',
    num_layers: 80,
    hidden_size: 8192,
    num_attention_heads: 64,
    num_kv_heads: 8,
    ffn_hidden_size: 28672,
    vocab_size: 32016,
    tied_embeddings: false,
    gated_mlp: true,
    base_num_parameters: 8000000000,
    precision: 'bf16',
  },
  custom_fields: {},
};

export const entityStorePromptTunedModel1: ModelEntity = {
  id: 'model-prompt-1',
  created_at: '2025-01-07T16:08:49.525322',
  updated_at: '2025-01-07T16:08:49.525324',
  name: 'some-custom-model',
  workspace: 'default',
  description: 'None',
  spec: {
    checkpoint_model_name: 'model',
    family: 'llama',
    num_layers: 80,
    hidden_size: 8192,
    num_attention_heads: 64,
    num_kv_heads: 8,
    ffn_hidden_size: 28672,
    vocab_size: 32016,
    tied_embeddings: false,
    gated_mlp: true,
    base_num_parameters: 8000000000,
    precision: 'bf16',
  },
  base_model: 'default/codellama-70b',
  project: '',
  custom_fields: {},
};

export const entityStoreCustomizedModel1: ModelEntity = {
  id: 'model-customized-1',
  created_at: '2025-01-07T16:08:49.525322',
  updated_at: '2025-01-07T16:08:49.525324',
  name: 'codellama-70b-dataset-AnxDxZ6MBzFprTU78BAYP9-lora',
  workspace: 'test',
  description: 'None',
  spec: {
    checkpoint_model_name: 'model',
    family: 'llama',
    num_layers: 80,
    hidden_size: 8192,
    num_attention_heads: 64,
    num_kv_heads: 8,
    ffn_hidden_size: 28672,
    vocab_size: 32016,
    tied_embeddings: false,
    gated_mlp: true,
    base_num_parameters: 8000000000,
    precision: 'bf16',
    context_size: 4096,
    num_virtual_tokens: 0,
    is_chat: true,
  },
  base_model: 'meta/codellama-70b',
  project: '',
  custom_fields: {},
};

export const emptyModelOutputPage: ModelEntitysPage = {
  data: [],
};

export const mixedModelOutputPage: ModelEntitysPage = {
  data: [entityStoreBaseModel1, entityStorePromptTunedModel1, entityStoreCustomizedModel1],
};

export const getEntityStoreLlmModels = (query?: ModelsListModelsParams) => {
  const models = [entityStoreBaseModel1, entityStorePromptTunedModel1, entityStoreCustomizedModel1];

  let data = models;

  if (query?.sort) {
    switch (query.sort) {
      case '-created_at':
        data = models.sort(
          (a, b) => new Date(b.created_at!).getTime() - new Date(a.created_at!).getTime()
        );
        break;
      case 'name':
        data = models.sort((a, b) => a.name!.localeCompare(b.name!));
        break;
      case '-name':
        data = models.sort((a, b) => b.name!.localeCompare(a.name!));
        break;
      // created_at ASC is the default
      case 'created_at':
      default:
        data = models.sort(
          (a, b) => new Date(a.created_at!).getTime() - new Date(b.created_at!).getTime()
        );
        break;
    }
  }
  return {
    data,
  };
};
