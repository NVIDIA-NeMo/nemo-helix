// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelWorkspaceGroup } from '@nemo/common/src/api/models/useModels';
import type {
  CreateJobRequest as DataDesignerJobRequest,
  DataDesignerConfig,
} from '@nemo/sdk/generated/data-designer/schema';
import {
  ERROR_NO_COLUMNS,
  ERROR_NO_LOADABLE_COLUMNS,
  getGeneratedJobRequestFromState,
  seedFromJobRequest,
  validateGeneratedJobRequest,
} from '@studio/routes/DataDesignerJobBuildRoute/aiSeed';

const MODEL_GROUPS: ModelWorkspaceGroup[] = [
  {
    workspace: 'default',
    models: [
      {
        id: 'default/llama-3.3',
        workspace: 'default',
        name: 'llama-3.3',
        created_at: '',
        updated_at: '',
        model_providers: ['default/nim'],
      },
    ],
  } as unknown as ModelWorkspaceGroup,
];

const config = (overrides: Partial<DataDesignerConfig> = {}): DataDesignerConfig =>
  ({
    columns: [
      {
        name: 'topic',
        column_type: 'sampler',
        sampler_type: 'category',
        params: { values: ['science', 'history'] },
      },
      {
        name: 'question',
        column_type: 'llm-text',
        prompt: 'Write a question about {{ topic }}.',
        model_alias: 'gen',
      },
    ],
    model_configs: [{ alias: 'gen', model: 'gpt-4o', provider: 'openai' }],
    ...overrides,
  }) as unknown as DataDesignerConfig;

const jobRequest = (overrides: Partial<DataDesignerConfig> = {}): DataDesignerJobRequest => ({
  name: 'qa-pairs',
  spec: { num_records: 50, config: config(overrides) },
});

describe('validateGeneratedJobRequest', () => {
  it('accepts a config the builder can load, resolving models against the workspace', () => {
    const result = validateGeneratedJobRequest(jobRequest(), MODEL_GROUPS);

    expect(result.status).toBe('valid');
    if (result.status !== 'valid') return;
    expect(result.seed.columns.map((column) => column.name)).toEqual(['topic', 'question']);
    expect(result.seed.models).toEqual([
      expect.objectContaining({
        alias: 'gen',
        model: 'default/llama-3.3',
        provider: 'default/nim',
      }),
    ]);
    expect(result.jobRequest.spec.config.model_configs?.[0]).toEqual(
      expect.objectContaining({ model: 'default/llama-3.3', provider: 'default/nim' })
    );
    expect(result.warnings).toEqual([
      '"gpt-4o" is not available in this workspace — alias "gen" now uses "default/llama-3.3".',
    ]);
  });

  it('points every alias at the model that drafted the config', () => {
    const result = validateGeneratedJobRequest(jobRequest(), MODEL_GROUPS, {
      model: 'default/llama-3.3',
      provider: 'default/nim',
    });

    expect(result.status).toBe('valid');
    if (result.status !== 'valid') return;
    expect(result.seed.models).toEqual([
      expect.objectContaining({
        alias: 'gen',
        model: 'default/llama-3.3',
        provider: 'default/nim',
      }),
    ]);
    expect(result.warnings).toEqual([]);
  });

  it('uses the generation model even when it is not in the loaded model list', () => {
    // The picker's list is search-filtered, so the selected model may not appear in it.
    const result = validateGeneratedJobRequest(jobRequest(), [], {
      model: 'default/some-other-model',
      provider: 'default/nim',
    });

    expect(result.status).toBe('valid');
    if (result.status !== 'valid') return;
    expect(result.seed.models[0]).toEqual(
      expect.objectContaining({ model: 'default/some-other-model', provider: 'default/nim' })
    );
  });

  it('says nothing when the draft already named the generation model', () => {
    const result = validateGeneratedJobRequest(
      jobRequest({
        model_configs: [{ alias: 'gen', model: 'default/llama-3.3', provider: 'default/nim' }],
      }),
      MODEL_GROUPS,
      { model: 'default/llama-3.3', provider: 'default/nim' }
    );

    expect(result.warnings).toEqual([]);
  });

  it('leaves generated models untouched when the workspace model list is empty', () => {
    const result = validateGeneratedJobRequest(jobRequest(), []);

    expect(result.status).toBe('valid');
    if (result.status !== 'valid') return;
    expect(result.seed.models[0].model).toBe('gpt-4o');
    expect(result.warnings).toEqual([]);
  });

  it('rejects a config with no columns', () => {
    const result = validateGeneratedJobRequest(jobRequest({ columns: [] }), MODEL_GROUPS);

    expect(result).toEqual({ status: 'invalid', errors: [ERROR_NO_COLUMNS], warnings: [] });
  });

  it('rejects a config whose columns are all unknown to the builder', () => {
    const result = validateGeneratedJobRequest(
      jobRequest({
        columns: [{ name: 'mystery', column_type: 'not-a-column-type' }],
        model_configs: [],
      } as unknown as Partial<DataDesignerConfig>),
      MODEL_GROUPS
    );

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') return;
    expect(result.errors).toContain(ERROR_NO_LOADABLE_COLUMNS);
  });

  it('adds a model config for an alias no model config defines', () => {
    const result = validateGeneratedJobRequest(
      jobRequest({ model_configs: [{ alias: 'other', model: 'gpt-4o', provider: 'openai' }] }),
      MODEL_GROUPS,
      { model: 'default/drafting-model', provider: 'default/nim' }
    );

    expect(result.status).toBe('valid');
    if (result.status !== 'valid') return;
    expect(result.jobRequest.spec.config.model_configs?.map((config) => config.alias)).toEqual([
      'other',
      'gen',
    ]);
  });

  it('rejects a column missing a required field', () => {
    const result = validateGeneratedJobRequest(
      jobRequest({
        columns: [{ name: 'question', column_type: 'llm-text', model_alias: 'gen' }],
      } as unknown as Partial<DataDesignerConfig>),
      MODEL_GROUPS
    );

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') return;
    expect(result.errors).toContain('question: Prompt is required.');
  });

  it('warns about columns it had to skip but still accepts the rest', () => {
    const result = validateGeneratedJobRequest(
      jobRequest({
        columns: [...config().columns, { name: 'mystery', column_type: 'not-a-column-type' }],
      } as unknown as Partial<DataDesignerConfig>),
      MODEL_GROUPS
    );

    expect(result.status).toBe('valid');
    expect(result.warnings).toContain(
      "Skipped 1 column(s) the builder can't edit: mystery (not-a-column-type)."
    );
  });

  it('defaults a non-positive record count and says so', () => {
    const request = jobRequest();
    request.spec.num_records = 0;

    const result = validateGeneratedJobRequest(request, MODEL_GROUPS);

    expect(result.status).toBe('valid');
    if (result.status !== 'valid') return;
    expect(result.seed.rows).toBe('100');
    expect(result.warnings).toContain(
      'Record count was not a positive whole number — defaulted to 100.'
    );
  });
});

describe('seedFromJobRequest', () => {
  it('falls back to a default name when the request has none', () => {
    const request = jobRequest();
    delete request.name;

    expect(seedFromJobRequest(request).name).toBe('untitled-dataset');
  });
});

describe('getGeneratedJobRequestFromState', () => {
  it('returns the request when the state carries one', () => {
    const request = jobRequest();

    expect(getGeneratedJobRequestFromState({ generatedJobRequest: request })).toBe(request);
  });

  it.each([[null], [undefined], [{}], [{ cloneJobRequest: {} }], [{ generatedJobRequest: {} }]])(
    'returns null for unrelated state (%o)',
    (state) => {
      expect(getGeneratedJobRequestFromState(state)).toBeNull();
    }
  );
});
