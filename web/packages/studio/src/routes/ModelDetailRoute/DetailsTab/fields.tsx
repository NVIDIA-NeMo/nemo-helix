// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parametersToString } from '@nemo/common/src/components/ModelDetailsTooltip/utils';
import { formatAbsoluteTimestamp } from '@nemo/common/src/components/RelativeTime/util';
import type { StatTileDiagnosticProps } from '@nemo/common/src/components/StatTile';
import { formatFinetuningType, kebabCaseToTitleCase } from '@nemo/common/src/utils/formatters';
import type {
  Adapter,
  InferenceParams,
  ModelEntity,
  ModelSpec,
} from '@nemo/sdk/generated/platform/schema';
import { formatMaybe } from '@studio/util/intakeTelemetry';

export type DetailValue = string | number | boolean | null | undefined;

export interface DetailField {
  label: string;
  value: DetailValue;
}

export type DetailTile = Pick<StatTileDiagnosticProps, 'label' | 'value' | 'trailingLabel'>;

type TileInput = Omit<DetailTile, 'value'> & { value: DetailValue };

export const hasDetailValue = (value: DetailValue) => value != null && value !== '';

/** Tiles without a value are dropped rather than shown as dashes. */
const presentTiles = (tiles: TileInput[]): DetailTile[] =>
  tiles
    .filter((tile) => hasDetailValue(tile.value))
    .map((tile) => ({ ...tile, value: formatMaybe(tile.value) }));

/** The model's size and shape at a glance. */
export const getModelTiles = (model: ModelEntity): DetailTile[] => {
  const spec = model.spec;
  return presentTiles([
    {
      label: 'Parameters',
      value: spec?.base_num_parameters
        ? parametersToString(spec.base_num_parameters, { format: 'short' })
        : undefined,
    },
    {
      label: 'Context Size',
      value: spec?.context_size
        ? parametersToString(spec.context_size, { format: 'short' })
        : undefined,
      trailingLabel: 'tokens',
    },
    { label: 'Precision', value: spec?.precision },
  ]);
};

const getModelType = (spec: ModelSpec | undefined) => {
  if (spec?.is_embedding_model || spec?.head_type === 'embedding') return 'Embedding';
  if (spec?.head_type === 'cross_encoder') return 'Reranker';
  if (spec?.is_chat) return 'Chat';
  if (spec?.head_type === 'causal_lm') return 'Text Generation';
  return undefined;
};

export const getModelFields = (model: ModelEntity): DetailField[] => {
  const spec = model.spec;
  const toolCalling = spec?.tool_call_config;
  const createdBy = model.ownership?.created_by;
  return [
    { label: 'Model Type', value: getModelType(spec) },
    { label: 'Family', value: spec?.family ? kebabCaseToTitleCase(spec.family) : undefined },
    { label: 'Checkpoint', value: spec?.checkpoint_model_name },
    {
      label: 'Tool Calling',
      value: toolCalling && (toolCalling.tool_call_parser ?? 'Supported'),
    },
    { label: 'Min GPUs to Fine-tune', value: spec?.minimum_gpus_all_weights },
    { label: 'Min GPUs for LoRA', value: spec?.minimum_gpus_lora },
    // Only worth calling out when set: it lets the model run code from its repository.
    { label: 'Trust Remote Code', value: model.trust_remote_code || undefined },
    { label: 'Created', value: model.created_at && formatAbsoluteTimestamp(model.created_at) },
    { label: 'Owner', value: typeof createdBy === 'string' ? createdBy : undefined },
  ];
};

export const getAdapterTiles = (adapter: Adapter): DetailTile[] => {
  return presentTiles([
    { label: 'Type', value: formatFinetuningType(adapter.finetuning_type) },
    { label: 'LoRA Rank', value: adapter.lora_config?.rank },
    { label: 'LoRA Alpha', value: adapter.lora_config?.alpha },
  ]);
};

export const getAdapterFields = (adapter: Adapter): DetailField[] => [
  { label: 'Enabled', value: adapter.enabled },
  { label: 'Created', value: adapter.created_at && formatAbsoluteTimestamp(adapter.created_at) },
  { label: 'Updated', value: adapter.updated_at && formatAbsoluteTimestamp(adapter.updated_at) },
];

export const getArchitectureTiles = (spec: ModelSpec): DetailTile[] => {
  return presentTiles([
    { label: 'Layers', value: spec.num_layers },
    { label: 'Hidden Size', value: spec.hidden_size },
    { label: 'Attention Heads', value: spec.num_attention_heads },
  ]);
};

/** Layer-level shape of the network, beyond the headline tiles. */
export const getArchitectureFields = (spec: ModelSpec): DetailField[] => [
  { label: 'FFN Hidden Size', value: spec.ffn_hidden_size },
  { label: 'KV Heads', value: spec.num_kv_heads },
  { label: 'Vocabulary Size', value: spec.vocab_size },
  { label: 'Tied Embeddings', value: spec.tied_embeddings },
  { label: 'Gated MLP', value: spec.gated_mlp },
  { label: 'Experts', value: spec.moe_config?.num_experts },
  { label: 'Experts per Token', value: spec.moe_config?.num_experts_per_tok },
  { label: 'Shared Experts', value: spec.moe_config?.num_shared_experts },
];

/** The sampling defaults applied when a request doesn't set its own. */
export const getInferenceDefaultFields = (params: InferenceParams | undefined): DetailField[] => [
  { label: 'Temperature', value: params?.temperature },
  { label: 'Top P', value: params?.top_p },
  { label: 'Max Tokens', value: params?.max_completion_tokens ?? params?.max_tokens },
  { label: 'Stop Sequences', value: params?.stop?.join(', ') },
];
