// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import {
  getAdapterTiles,
  getArchitectureFields,
  getArchitectureTiles,
  getModelTiles,
} from '@studio/routes/ModelDetailRoute/DetailsTab/fields';

const model = {
  name: 'my-model',
  workspace: 'ws',
  spec: {
    base_num_parameters: 2_000_000_000,
    context_size: 32_768,
    precision: 'bfloat16',
    num_layers: 28,
    hidden_size: 2048,
    tied_embeddings: true,
  },
} as ModelEntity;

describe('getModelTiles', () => {
  it('formats the headline stats', () => {
    expect(getModelTiles(model)).toEqual([
      { label: 'Parameters', value: '2B' },
      expect.objectContaining({ label: 'Context Size', trailingLabel: 'tokens' }),
      { label: 'Precision', value: 'bfloat16' },
    ]);
  });

  it('drops tiles without a value', () => {
    expect(getModelTiles({ ...model, spec: undefined })).toEqual([]);
  });
});

describe('getAdapterTiles', () => {
  it('shows the type and LoRA settings', () => {
    const adapter = {
      name: 'a',
      finetuning_type: 'lora',
      lora_config: { rank: 16, alpha: 32 },
    } as Adapter;

    expect(getAdapterTiles(adapter).map((tile) => [tile.label, tile.value])).toEqual([
      ['Type', 'LoRA'],
      ['LoRA Rank', '16'],
      ['LoRA Alpha', '32'],
    ]);
  });
});

describe('architecture', () => {
  it('splits headline stats from the remaining fields', () => {
    const spec = model.spec!;

    expect(getArchitectureTiles(spec).map((tile) => tile.label)).toEqual(['Layers', 'Hidden Size']);
    expect(getArchitectureFields(spec)).toContainEqual({ label: 'Tied Embeddings', value: 'Yes' });
  });
});
