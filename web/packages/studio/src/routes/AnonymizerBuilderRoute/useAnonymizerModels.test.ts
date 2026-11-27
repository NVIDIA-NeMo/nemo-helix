// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelProvider } from '@nemo/sdk/generated/platform/schema';
import { resolveServedModel } from '@studio/routes/AnonymizerBuilderRoute/useAnonymizerModels';

const makeProvider = (name: string, served: Record<string, string>): ModelProvider =>
  ({
    workspace: 'ws',
    name,
    served_models: Object.entries(served).map(([model_entity_id, served_model_name]) => ({
      model_entity_id,
      served_model_name,
    })),
  }) as ModelProvider;

const providers = [
  makeProvider('a', { 'ws/llama': 'llama-on-a' }),
  makeProvider('b', {
    'ws/llama': 'llama-on-b',
    'ws/llama&adapters/ws/tuned': 'tuned-on-b',
  }),
];

describe('resolveServedModel', () => {
  it('takes the first provider serving the model', () => {
    expect(resolveServedModel(providers, 'ws/llama')).toEqual({
      provider: 'ws/a',
      servedModelName: 'llama-on-a',
    });
  });

  it('prefers a provider the user filtered by, with its own served name', () => {
    expect(resolveServedModel(providers, 'ws/llama', ['ws/b'])).toEqual({
      provider: 'ws/b',
      servedModelName: 'llama-on-b',
    });
  });

  it('falls back to a serving provider when the filtered one does not serve the model', () => {
    expect(resolveServedModel(providers, 'ws/llama', ['ws/other'])?.provider).toBe('ws/a');
  });

  it('resolves an adapter by its composite id', () => {
    expect(resolveServedModel(providers, 'ws/llama&adapters/ws/tuned')).toEqual({
      provider: 'ws/b',
      servedModelName: 'tuned-on-b',
    });
  });

  it('returns nothing for a model no provider serves', () => {
    expect(resolveServedModel(providers, 'ws/unknown')).toBeUndefined();
  });
});
