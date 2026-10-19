// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parametersToString } from '@nemo/common/src/components/ModelDetailsTooltip/utils';
import { formatAbsoluteTimestamp } from '@nemo/common/src/components/RelativeTime/util';
import type { StatTileDiagnosticProps } from '@nemo/common/src/components/StatTile';
import { formatFinetuningType, kebabCaseToTitleCase } from '@nemo/common/src/utils/formatters';
import type { Adapter, ModelEntity, ModelSpec } from '@nemo/sdk/generated/platform/schema';
import type { ReactNode } from 'react';

export interface DetailField {
  label: string;
  value: ReactNode;
}

export type DetailTile = Pick<StatTileDiagnosticProps, 'label' | 'value' | 'trailingLabel'>;

const formatCount = (value: number | undefined) =>
  value == null ? undefined : value.toLocaleString();

const formatBoolean = (value: boolean | undefined) =>
  value == null ? undefined : value ? 'Yes' : 'No';

const formatTimestamp = (value: string | undefined) =>
  value ? formatAbsoluteTimestamp(value) : undefined;

type TileInput = Omit<DetailTile, 'value'> & { value: string | undefined };

/** Tiles without a value are dropped. */
const presentTiles = (tiles: TileInput[]): DetailTile[] =>
  tiles.filter((tile): tile is DetailTile => Boolean(tile.value));

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

export const getModelFields = (model: ModelEntity): DetailField[] => {
  const spec = model.spec;
  const createdBy = model.ownership?.created_by;
  return [
    { label: 'Base Model', value: model.base_model },
    {
      label: 'Fine-tuning Type',
      value: model.finetuning_type ? formatFinetuningType(model.finetuning_type) : undefined,
    },
    { label: 'Family', value: spec?.family ? kebabCaseToTitleCase(spec.family) : undefined },
    { label: 'Chat', value: spec?.is_chat ? 'Supported' : undefined },
    { label: 'Checkpoint', value: spec?.checkpoint_model_name },
    { label: 'Created', value: formatTimestamp(model.created_at) },
    { label: 'Updated', value: formatTimestamp(model.updated_at) },
    { label: 'Owner', value: createdBy ? String(createdBy) : undefined },
  ];
};

export const getAdapterTiles = (adapter: Adapter): DetailTile[] => {
  return presentTiles([
    { label: 'Type', value: formatFinetuningType(adapter.finetuning_type) },
    { label: 'LoRA Rank', value: formatCount(adapter.lora_config?.rank) },
    { label: 'LoRA Alpha', value: formatCount(adapter.lora_config?.alpha) },
  ]);
};

export const getAdapterFields = (adapter: Adapter): DetailField[] => [
  { label: 'Enabled', value: formatBoolean(adapter.enabled) },
  { label: 'Created', value: formatTimestamp(adapter.created_at) },
  { label: 'Updated', value: formatTimestamp(adapter.updated_at) },
];

export const getArchitectureTiles = (spec: ModelSpec): DetailTile[] => {
  return presentTiles([
    { label: 'Layers', value: formatCount(spec.num_layers) },
    { label: 'Hidden Size', value: formatCount(spec.hidden_size) },
    { label: 'Attention Heads', value: formatCount(spec.num_attention_heads) },
  ]);
};

/** Layer-level shape of the network, beyond the headline tiles. */
export const getArchitectureFields = (spec: ModelSpec): DetailField[] => [
  { label: 'FFN Hidden Size', value: formatCount(spec.ffn_hidden_size) },
  { label: 'KV Heads', value: formatCount(spec.num_kv_heads) },
  { label: 'Vocabulary Size', value: formatCount(spec.vocab_size) },
  { label: 'Tied Embeddings', value: formatBoolean(spec.tied_embeddings) },
  { label: 'Gated MLP', value: formatBoolean(spec.gated_mlp) },
  { label: 'Min GPUs (Full Weights)', value: formatCount(spec.minimum_gpus_all_weights) },
  { label: 'Min GPUs (LoRA)', value: formatCount(spec.minimum_gpus_lora) },
];
