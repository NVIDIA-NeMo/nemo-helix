// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelWorkspaceGroup } from '@nemo/common/src/api/models/useModels';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { DEFAULT_BUILD_MODEL_NAME } from '@studio/constants/constants';
import { resolveDefaultGuardrailModel } from '@studio/routes/guardrails/defaultModel';

const model = (name: string, providers: string[] = ['default/nim']): ModelEntity =>
  ({ name, workspace: 'default', model_providers: providers }) as ModelEntity;

const groups = (...models: ModelEntity[]): ModelWorkspaceGroup[] => [
  { workspace: 'default', models } as ModelWorkspaceGroup,
];

describe('resolveDefaultGuardrailModel', () => {
  it('returns null when there are no models', () => {
    expect(resolveDefaultGuardrailModel([])).toBeNull();
  });

  it('returns null when every model lacks a provider', () => {
    expect(resolveDefaultGuardrailModel(groups(model('llama', [])))).toBeNull();
  });

  it('prefers the default build model over an alphabetically earlier one', () => {
    const result = resolveDefaultGuardrailModel(
      groups(model('adept-fuyu-8b'), model(DEFAULT_BUILD_MODEL_NAME))
    );
    expect(result).toBe(`default/${DEFAULT_BUILD_MODEL_NAME}`);
  });

  it('prefers the requested model by bare name', () => {
    const result = resolveDefaultGuardrailModel(
      groups(model('other'), model('meta-llama-3-2-1b-instruct')),
      'other-workspace/meta-llama-3-2-1b-instruct'
    );
    expect(result).toBe('default/meta-llama-3-2-1b-instruct');
  });

  it('falls back to the first usable model', () => {
    const result = resolveDefaultGuardrailModel(groups(model('other')), 'meta/absent');
    expect(result).toBe('default/other');
  });

  it('skips provider-less models when falling back', () => {
    const result = resolveDefaultGuardrailModel(
      groups(model('no-provider', []), model('usable')),
      'meta/absent'
    );
    expect(result).toBe('default/usable');
  });
});
