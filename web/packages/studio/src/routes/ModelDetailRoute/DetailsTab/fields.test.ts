// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import {
  getAdapterTiles,
  getArchitectureFields,
  getArchitectureTiles,
  getModelFields,
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

describe('getModelFields', () => {
  const valueOf = (entity: ModelEntity, label: string) =>
    getModelFields(entity).find((field) => field.label === label)?.value;

  it('names the model type from its spec', () => {
    expect(valueOf({ ...model, spec: { ...model.spec!, is_chat: true } }, 'Model Type')).toBe(
      'Chat'
    );
    expect(
      valueOf({ ...model, spec: { ...model.spec!, head_type: 'cross_encoder' } }, 'Model Type')
    ).toBe('Reranker');
  });

  it('only flags trusted remote code when it is on', () => {
    expect(valueOf(model, 'Trust Remote Code')).toBeUndefined();
    expect(valueOf({ ...model, trust_remote_code: true }, 'Trust Remote Code')).toBe(true);
  });

  it('ignores an owner that is not a string', () => {
    expect(valueOf({ ...model, ownership: { created_by: { id: 1 } } }, 'Owner')).toBeUndefined();
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
    expect(getArchitectureFields(spec)).toContainEqual({ label: 'Tied Embeddings', value: true });
  });
});
