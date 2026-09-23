// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type Adapter,
  FinetuningType,
  ModelDeploymentStatus,
  type ModelEntity,
} from '@nemo/sdk/generated/platform/schema';

import {
  getModelEntityChatStatus,
  groupModelsByWorkspace,
  isServedByBaseModel,
  toInferenceModelEntityId,
  toInferenceModelName,
} from './models';

const createModel = (overrides: Partial<ModelEntity> = {}): ModelEntity => ({
  id: 'test-id',
  name: 'test-model',
  workspace: 'test-namespace',
  created_at: '2025-01-07T12:00:00Z',
  updated_at: '2025-01-07T12:00:00Z',
  ...overrides,
});

const createAdapter = (overrides: Partial<Adapter> = {}): Adapter => ({
  name: 'test-adapter',
  workspace: 'test-namespace',
  fileset: 'test-namespace/test-adapter',
  finetuning_type: FinetuningType.lora,
  ...overrides,
});

describe('toInferenceModelName', () => {
  it('returns the bare model name when there is no adapter', () => {
    const model = createModel({ workspace: 'default', name: 'qwen3-0.6b' });
    expect(toInferenceModelName(model)).toBe('qwen3-0.6b');
    expect(toInferenceModelName(model, null)).toBe('qwen3-0.6b');
  });

  // The gateway splits on "&adapters/" and looks the VirtualModel up by the base
  // segment, taking the workspace from the URL. A workspace-qualified base segment
  // would key a VM that does not exist.
  it('leaves the base segment unqualified so the gateway can key the base VM', () => {
    const model = createModel({ workspace: 'default', name: 'qwen3-0.6b' });
    const adapter = createAdapter({ workspace: 'default', name: 'pirate-voice' });
    expect(toInferenceModelName(model, adapter)).toBe('qwen3-0.6b&adapters/default/pirate-voice');
  });

  it('qualifies the adapter segment, which may live in another workspace', () => {
    const model = createModel({ workspace: 'nvidia', name: 'qwen3-0.6b' });
    const adapter = createAdapter({ workspace: 'default', name: 'pirate-voice' });
    expect(toInferenceModelName(model, adapter)).toBe('qwen3-0.6b&adapters/default/pirate-voice');
  });
});

describe('toInferenceModelEntityId', () => {
  it('returns the plain entity ref when there is no adapter', () => {
    const model = createModel({ workspace: 'default', name: 'qwen3-0.6b' });
    expect(toInferenceModelEntityId(model)).toBe('default/qwen3-0.6b');
    expect(toInferenceModelEntityId(model, null)).toBe('default/qwen3-0.6b');
  });

  it('builds the composite id the provider reconciler writes for an adapter', () => {
    const model = createModel({ workspace: 'default', name: 'qwen3-0.6b' });
    const adapter = createAdapter({ workspace: 'default', name: 'pirate-voice' });
    expect(toInferenceModelEntityId(model, adapter)).toBe(
      'default/qwen3-0.6b&adapters/default/pirate-voice'
    );
  });

  it('carries the adapter workspace, which may differ from the base model workspace', () => {
    const model = createModel({ workspace: 'nvidia', name: 'qwen3-0.6b' });
    const adapter = createAdapter({ workspace: 'default', name: 'pirate-voice' });
    expect(toInferenceModelEntityId(model, adapter)).toBe(
      'nvidia/qwen3-0.6b&adapters/default/pirate-voice'
    );
  });

  it('matches a served_models entry by identity', () => {
    // This id exists to look up `provider.served_models`; the value that goes on
    // the wire is that entry's `served_model_name`, never this.
    const model = createModel({ workspace: 'default', name: 'qwen3-0.6b' });
    const adapter = createAdapter({ workspace: 'default', name: 'pirate-voice' });
    const servedModels = [
      { model_entity_id: 'default/qwen3-0.6b', served_model_name: 'default/qwen3-0.6b' },
      {
        model_entity_id: 'default/qwen3-0.6b&adapters/default/pirate-voice',
        served_model_name: 'default--pirate-voice',
      },
    ];
    const match = servedModels.find(
      (sm) => sm.model_entity_id === toInferenceModelEntityId(model, adapter)
    );
    expect(match?.served_model_name).toBe('default--pirate-voice');
  });
});

describe('getModelEntityChatStatus', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('returns pending when model created less than 5 minutes ago', () => {
    const now = new Date('2025-01-07T12:00:00Z').getTime();
    vi.setSystemTime(now);

    const model = createModel({
      created_at: '2025-01-07T11:57:00Z', // 3 minutes ago
    });

    expect(getModelEntityChatStatus(model)).toBe('pending');
  });

  it('skips the grace period for a freshly created model that already has an api_endpoint', () => {
    // Autodiscovered from an external provider (e.g. build.nvidia.com): the endpoint is
    // live the moment the entity exists, so there is no deployment to wait for.
    vi.setSystemTime(new Date('2025-01-07T12:00:00Z').getTime());

    const model = createModel({
      created_at: '2025-01-07T11:57:00Z', // 3 minutes ago
      api_endpoint: { url: 'https://integrate.api.nvidia.com/v1/chat/completions' },
    });

    expect(getModelEntityChatStatus(model)).toBe('enabled');
  });

  it('still applies the grace period to a fresh model with no api_endpoint', () => {
    // Deployment-backed: the artifact genuinely is still being populated.
    vi.setSystemTime(new Date('2025-01-07T12:00:00Z').getTime());

    const model = createModel({ created_at: '2025-01-07T11:57:00Z' });

    expect(getModelEntityChatStatus(model, { deploymentStatus: null })).toBe('pending');
  });

  it('still reports pending while the deployment chain is loading, endpoint or not', () => {
    vi.setSystemTime(new Date('2025-01-07T12:00:00Z').getTime());

    const model = createModel({
      created_at: '2025-01-07T11:57:00Z',
      api_endpoint: { url: 'https://integrate.api.nvidia.com/v1/chat/completions' },
    });

    expect(getModelEntityChatStatus(model, { deploymentLoading: true })).toBe('pending');
  });

  it('returns enabled when model created more than 5 minutes ago', () => {
    const now = new Date('2025-01-07T12:00:00Z').getTime();
    vi.setSystemTime(now);

    const model = createModel({
      created_at: '2025-01-07T11:50:00Z', // 10 minutes ago
    });

    expect(getModelEntityChatStatus(model)).toBe('enabled');
  });

  it('returns enabled when no created_at field', () => {
    const model = createModel();

    expect(getModelEntityChatStatus(model)).toBe('enabled');
  });

  it('handles created_at without timezone indicator (no Z suffix) as UTC', () => {
    const now = new Date('2025-01-07T12:00:00Z').getTime();
    vi.setSystemTime(now);

    const model = createModel({
      created_at: '2025-01-07T11:57:00', // No Z suffix, 3 minutes ago
    });

    expect(getModelEntityChatStatus(model)).toBe('pending');
  });

  it('handles created_at without timezone when model is old', () => {
    const now = new Date('2025-01-07T12:00:00Z').getTime();
    vi.setSystemTime(now);

    const model = createModel({
      created_at: '2025-01-07T11:50:00', // No Z suffix, 10 minutes ago
    });

    expect(getModelEntityChatStatus(model)).toBe('enabled');
  });

  it('handles created_at with +00:00 timezone offset', () => {
    const now = new Date('2025-01-07T12:00:00Z').getTime();
    vi.setSystemTime(now);

    // 11:57:00+00:00 is the same as 11:57:00Z (3 minutes ago)
    const model = createModel({
      created_at: '2025-01-07T11:57:00+00:00',
    });

    expect(getModelEntityChatStatus(model)).toBe('pending');
  });

  it('handles created_at with positive timezone offset', () => {
    const now = new Date('2025-01-07T12:00:00Z').getTime();
    vi.setSystemTime(now);

    // 17:27:00+05:30 is equivalent to 11:57:00 UTC (3 minutes ago)
    const model = createModel({
      created_at: '2025-01-07T17:27:00+05:30',
    });

    expect(getModelEntityChatStatus(model)).toBe('pending');
  });

  it('handles created_at with negative timezone offset', () => {
    const now = new Date('2025-01-07T12:00:00Z').getTime();
    vi.setSystemTime(now);

    // 06:57:00-05:00 is equivalent to 11:57:00 UTC (3 minutes ago)
    const model = createModel({
      created_at: '2025-01-07T06:57:00-05:00',
    });

    expect(getModelEntityChatStatus(model)).toBe('pending');
  });
});

describe('getModelEntityChatStatus (deployment & api_endpoint)', () => {
  const stableModel: ModelEntity = createModel({
    created_at: '2025-01-07T11:00:00Z',
    updated_at: '2025-01-07T11:00:00Z',
  });

  it('enables standalone models with api_endpoint.url', () => {
    expect(
      getModelEntityChatStatus({
        ...stableModel,
        api_endpoint: { url: 'https://example/v1/chat/completions' },
      } as ModelEntity)
    ).toBe('enabled');
  });

  it('disables standalone models without api_endpoint when deployment is null', () => {
    expect(getModelEntityChatStatus(stableModel, { deploymentStatus: null })).toBe('disabled');
  });

  it('enables standalone models with READY deployment and no api_endpoint', () => {
    expect(
      getModelEntityChatStatus(stableModel, { deploymentStatus: ModelDeploymentStatus.READY })
    ).toBe('enabled');
  });

  it('requires READY deployment for base_model when status is provided', () => {
    const derived = { ...stableModel, base_model: 'ws/parent' } as ModelEntity;
    expect(
      getModelEntityChatStatus(derived, { deploymentStatus: ModelDeploymentStatus.READY })
    ).toBe('enabled');
    expect(
      getModelEntityChatStatus(derived, { deploymentStatus: ModelDeploymentStatus.ERROR })
    ).toBe('disabled');
    expect(getModelEntityChatStatus(derived)).toBe('enabled');
  });

  it('returns pending while deployment is loading', () => {
    expect(getModelEntityChatStatus(stableModel, { deploymentLoading: true })).toBe('pending');
  });
});

describe('groupModelsByWorkspace', () => {
  it('returns [] for an empty list', () => {
    expect(groupModelsByWorkspace([])).toEqual([]);
  });

  it('groups models by workspace and preserves model order within each group', () => {
    const m1 = createModel({ id: 'm1', name: 'a', workspace: 'nvidia' });
    const m2 = createModel({ id: 'm2', name: 'b', workspace: 'meta' });
    const m3 = createModel({ id: 'm3', name: 'c', workspace: 'nvidia' });
    const groups = groupModelsByWorkspace([m1, m2, m3]);

    expect(groups).toHaveLength(2);
    const nvidia = groups.find((g) => g.workspace === 'nvidia');
    const meta = groups.find((g) => g.workspace === 'meta');
    expect(nvidia?.models).toEqual([m1, m3]);
    expect(meta?.models).toEqual([m2]);
  });

  it("falls back to 'default' when workspace is missing", () => {
    const m = {
      ...createModel({ name: 'a' }),
      workspace: undefined,
    } as unknown as ModelEntity;
    const [group] = groupModelsByWorkspace([m]);
    expect(group.workspace).toBe('default');
    expect(group.models).toEqual([m]);
  });

  it('sorts groups alphabetically by workspace name when sort is true', () => {
    const groups = groupModelsByWorkspace(
      [
        createModel({ id: 'm1', name: 'a', workspace: 'nvidia' }),
        createModel({ id: 'm2', name: 'b', workspace: 'meta' }),
        createModel({ id: 'm3', name: 'c', workspace: 'deepseek' }),
      ],
      { sort: true }
    );
    expect(groups.map((g) => g.workspace)).toEqual(['deepseek', 'meta', 'nvidia']);
  });
});

describe('isServedByBaseModel', () => {
  // The reported bug: a full-weight run registers its own model, deployment and
  // provider, but sets `base_model` like any fine-tune. Reading that as "served by
  // the base" resolved status against a base that is usually not deployed, so the
  // chat panel said "This model does not have an active deployment".
  it('is false for a full-weight fine-tune', () => {
    expect(
      isServedByBaseModel(
        createModel({
          base_model: 'default/qwen3-0.6b',
          finetuning_type: FinetuningType.all_weights,
        })
      )
    ).toBe(false);
  });

  // Merged weights are a complete model despite the `lora` in the name — which is
  // why the set is explicit rather than a prefix match.
  it('is false for a merged LoRA', () => {
    expect(
      isServedByBaseModel(
        createModel({
          base_model: 'default/qwen3-0.6b',
          finetuning_type: FinetuningType.lora_merged,
        })
      )
    ).toBe(false);
  });

  it('is false for a preference-optimisation run', () => {
    expect(
      isServedByBaseModel(
        createModel({ base_model: 'default/qwen3-0.6b', finetuning_type: FinetuningType.dpo })
      )
    ).toBe(false);
  });

  it.each([
    FinetuningType.lora,
    FinetuningType.qlora,
    FinetuningType.dora,
    FinetuningType.prompt_tuning,
    FinetuningType.p_tuning,
    FinetuningType.soft_prompt,
  ])('is true for %s, which has no weights of its own', (finetuning_type) => {
    expect(
      isServedByBaseModel(createModel({ base_model: 'default/qwen3-0.6b', finetuning_type }))
    ).toBe(true);
  });

  it('is false without a base model at all', () => {
    expect(isServedByBaseModel(createModel({ finetuning_type: FinetuningType.lora }))).toBe(false);
  });

  // Not a fine-tune: nothing says it is served by anything but itself.
  it('is false when the finetuning type is absent', () => {
    expect(isServedByBaseModel(createModel({ base_model: 'default/qwen3-0.6b' }))).toBe(false);
  });
});
