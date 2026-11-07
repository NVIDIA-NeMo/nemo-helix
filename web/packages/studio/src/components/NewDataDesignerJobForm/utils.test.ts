// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { COPY_NAME_SUFFIX } from '@nemo/common/src/utils/entityName';
import { getErrorMessage } from '@nemo/common/src/utils/error';
import type {
  CreateJob as DataDesignerJob,
  CreateJobRequest as DataDesignerJobRequest,
} from '@nemo/sdk/generated/data-designer/schema';
import {
  addMissingAliasConfigs,
  applyFormModelToJobRequest,
  buildClonedJobRequest,
  getCloneJobRequestFromState,
  getWorkspaceAndModel,
  modelsFromProviders,
  PARSE_ERROR_INVALID_JSON,
  PARSE_ERROR_MISSING_CONFIG,
  parseJsonContentToJobRequest,
  parseToolResponseToJobRequest,
  sanitizeJobRequestName,
} from '@studio/components/NewDataDesignerJobForm/utils';

describe('getErrorMessage', () => {
  it('returns message from an Error instance', () => {
    expect(getErrorMessage(new Error('boom'), 'default')).toBe('boom');
  });

  it('returns default when value is not an Error', () => {
    expect(getErrorMessage('string error', 'default')).toBe('default');
    expect(getErrorMessage(null, 'default')).toBe('default');
  });
});

describe('getWorkspaceAndModel', () => {
  it('parses workspace/name from slash-separated ref', () => {
    const result = getWorkspaceAndModel('ws1/my-model', 'fallback');
    expect(result.workspace).toBe('ws1');
    expect(result.name).toBe('my-model');
  });

  it('uses fallback workspace when no slash', () => {
    const result = getWorkspaceAndModel('my-model', 'fallback');
    expect(result).toEqual({ workspace: 'fallback', name: 'my-model' });
  });
});

describe('modelsFromProviders', () => {
  it('returns empty array for empty providers', () => {
    expect(modelsFromProviders([])).toEqual([]);
  });

  it('builds options from served_models', () => {
    const providers = [
      {
        workspace: 'ws',
        name: 'prov1',
        served_models: [{ model_entity_id: 'ws/model-a', served_model_name: 'model-a' }],
      },
    ];
    const result = modelsFromProviders(providers as never);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('ws/model-a');
    expect(result[0].served_model_name).toBe('model-a');
    expect(result[0].model_providers).toEqual(['ws/prov1']);
  });

  it('dedupes by id and merges providers', () => {
    const providers = [
      {
        workspace: 'ws',
        name: 'prov1',
        served_models: [{ model_entity_id: 'ws/model-a', served_model_name: 'model-a' }],
      },
      {
        workspace: 'ws',
        name: 'prov2',
        served_models: [{ model_entity_id: 'ws/model-a', served_model_name: 'model-a' }],
      },
    ];
    const result = modelsFromProviders(providers as never);
    expect(result).toHaveLength(1);
    expect(result[0].model_providers).toContain('ws/prov1');
    expect(result[0].model_providers).toContain('ws/prov2');
  });

  it('skips served_models without model_entity_id', () => {
    const providers = [
      {
        workspace: 'ws',
        name: 'prov1',
        served_models: [{ served_model_name: 'model-a' }],
      },
    ];
    const result = modelsFromProviders(providers as never);
    expect(result).toHaveLength(0);
  });
});

describe('sanitizeJobRequestName', () => {
  it('replaces spaces with hyphens', () => {
    const req = { name: 'my job name' } as DataDesignerJobRequest;
    expect(sanitizeJobRequestName(req).name).toBe('my-job-name');
  });

  it('collapses multiple hyphens', () => {
    const req = { name: 'my  job   name' } as DataDesignerJobRequest;
    expect(sanitizeJobRequestName(req).name).toBe('my-job-name');
  });

  it('trims leading/trailing hyphens', () => {
    const req = { name: ' -edge- ' } as DataDesignerJobRequest;
    expect(sanitizeJobRequestName(req).name).toBe('edge');
  });

  it('lowercases and drops characters the entity name pattern rejects', () => {
    const req = { name: 'Product-Reviews_2024!' } as DataDesignerJobRequest;
    expect(sanitizeJobRequestName(req).name).toBe('product-reviews_2024');
  });

  it('returns request unchanged when name is empty', () => {
    const req = { name: '' } as DataDesignerJobRequest;
    expect(sanitizeJobRequestName(req)).toBe(req);
  });

  it('returns request unchanged when name is undefined', () => {
    const req = {} as DataDesignerJobRequest;
    expect(sanitizeJobRequestName(req)).toBe(req);
  });
});

describe('parseToolResponseToJobRequest', () => {
  it('returns null for invalid JSON', () => {
    expect(parseToolResponseToJobRequest('not json')).toBeNull();
  });

  it('parses wrapped { job_request } format', () => {
    const input = JSON.stringify({ job_request: { spec: { config: {} } } });
    expect(parseToolResponseToJobRequest(input)).toEqual({ spec: { config: {} } });
  });

  it('parses direct spec-shaped payload', () => {
    const input = JSON.stringify({ spec: { config: {} } });
    expect(parseToolResponseToJobRequest(input)).toEqual({ spec: { config: {} } });
  });

  it('returns null for object without job_request or spec', () => {
    expect(parseToolResponseToJobRequest(JSON.stringify({ foo: 1 }))).toBeNull();
  });

  it('returns null for non-object JSON', () => {
    expect(parseToolResponseToJobRequest('"string"')).toBeNull();
  });
});

describe('parseJsonContentToJobRequest', () => {
  it('returns null/null for empty content', () => {
    expect(parseJsonContentToJobRequest('')).toEqual({ jobRequest: null, error: null });
    expect(parseJsonContentToJobRequest('   ')).toEqual({ jobRequest: null, error: null });
  });

  it('returns error for invalid JSON', () => {
    expect(parseJsonContentToJobRequest('{')).toEqual({
      jobRequest: null,
      error: PARSE_ERROR_INVALID_JSON,
    });
  });

  it('returns error when spec.config is missing', () => {
    expect(parseJsonContentToJobRequest(JSON.stringify({ foo: 1 }))).toEqual({
      jobRequest: null,
      error: PARSE_ERROR_MISSING_CONFIG,
    });
  });

  it('returns sanitized job request on valid input', () => {
    const input = JSON.stringify({ spec: { config: { columns: [] } }, name: 'my job' });
    const result = parseJsonContentToJobRequest(input);
    expect(result.error).toBeNull();
    expect(result.jobRequest?.name).toBe('my-job');
  });
});

describe('applyFormModelToJobRequest', () => {
  it('returns request unchanged when no model_configs', () => {
    const req = {
      spec: { config: { columns: [], model_configs: [] } },
    } as unknown as DataDesignerJobRequest;
    expect(applyFormModelToJobRequest(req, 'ref', 'prov', 'served')).toBe(req);
  });

  it('creates a model config when columns use an alias but model_configs is empty', () => {
    const req = {
      spec: {
        config: { columns: [{ model_alias: 'review-model', name: 'col1' }], model_configs: [] },
      },
    } as unknown as DataDesignerJobRequest;

    const result = applyFormModelToJobRequest(req, 'new-ref', 'new-prov', 'new-served');
    expect(result.spec?.config?.model_configs).toEqual([
      { alias: 'new-ref', model: 'new-served', provider: 'new-prov' },
    ]);
    expect(result.spec?.config?.columns[0]).toEqual(
      expect.objectContaining({ model_alias: 'new-ref' })
    );
  });

  it('applies model to model_configs and columns with model_alias', () => {
    const req = {
      spec: {
        config: {
          columns: [{ model_alias: 'old-ref', name: 'col1' }, { name: 'col2' }],
          model_configs: [{ alias: 'old', model: 'old-model', provider: 'old-prov' }],
        },
      },
    } as unknown as DataDesignerJobRequest;

    const result = applyFormModelToJobRequest(req, 'new-ref', 'new-prov', 'new-served');
    expect(result.spec?.config?.model_configs?.[0]).toEqual(
      expect.objectContaining({
        alias: 'new-ref',
        model: 'new-served',
        provider: 'new-prov',
      })
    );
    expect(result.spec?.config?.columns[0]).toEqual(
      expect.objectContaining({ model_alias: 'new-ref' })
    );
    // col without model_alias stays unchanged
    expect(result.spec?.config?.columns[1]).toEqual(expect.objectContaining({ name: 'col2' }));
    expect(result.spec?.config?.columns[1]).not.toHaveProperty('model_alias');
  });
});

describe('addMissingAliasConfigs', () => {
  type Columns = Parameters<typeof addMissingAliasConfigs>[1];

  it('copies the first config for every alias no config defines', () => {
    const configs = [
      {
        alias: 'writer',
        model: 'm',
        provider: 'p',
        inference_parameters: { generation_type: 'chat-completion' as const },
      },
    ];
    const columns = [
      { name: 'a', column_type: 'llm-text', model_alias: 'writer' },
      { name: 'b', column_type: 'llm-judge', model_alias: 'judge' },
      { name: 'c', column_type: 'llm-text', model_alias: 'judge' },
    ] as unknown as Columns;

    expect(addMissingAliasConfigs(configs, columns)).toEqual([
      configs[0],
      { ...configs[0], alias: 'judge' },
    ]);
  });

  it('creates empty model fields when there is no config to copy', () => {
    const columns = [
      { name: 'a', column_type: 'llm-text', model_alias: 'writer' },
    ] as unknown as Columns;
    expect(addMissingAliasConfigs([], columns)).toEqual([
      { alias: 'writer', model: '', provider: '' },
    ]);
  });
});

describe('buildClonedJobRequest', () => {
  const makeJob = (overrides: Partial<DataDesignerJob> = {}): DataDesignerJob =>
    ({
      name: 'reviews',
      description: 'Synthetic product reviews',
      spec: {
        job_config: {
          num_records: 250,
          config: { columns: [{ name: 'col1' }], model_configs: [] },
        },
        model_providers: [],
        model_configs: [],
      },
      ...overrides,
    }) as DataDesignerJob;

  it('copies the job config into a request with a -copy name', () => {
    const result = buildClonedJobRequest(makeJob());
    expect(result).toEqual({
      name: `reviews${COPY_NAME_SUFFIX}`,
      description: 'Synthetic product reviews',
      spec: {
        num_records: 250,
        config: { columns: [{ name: 'col1' }], model_configs: [] },
      },
    });
  });

  it('returns null when the job has no usable config', () => {
    expect(buildClonedJobRequest(makeJob({ spec: undefined }))).toBeNull();
    expect(
      buildClonedJobRequest(
        makeJob({
          spec: {
            job_config: { num_records: 1 },
            model_providers: [],
            model_configs: [],
          },
        } as unknown as Partial<DataDesignerJob>)
      )
    ).toBeNull();
  });
});

describe('getCloneJobRequestFromState', () => {
  const cloneJobRequest: DataDesignerJobRequest = {
    name: 'reviews-copy',
    spec: { num_records: 10, config: { columns: [], model_configs: [] } },
  };

  it('returns the cloned job request from valid state', () => {
    expect(getCloneJobRequestFromState({ cloneJobRequest })).toBe(cloneJobRequest);
  });

  it('returns null for absent or malformed state', () => {
    expect(getCloneJobRequestFromState(null)).toBeNull();
    expect(getCloneJobRequestFromState(undefined)).toBeNull();
    expect(getCloneJobRequestFromState({})).toBeNull();
    expect(getCloneJobRequestFromState('nope')).toBeNull();
    expect(getCloneJobRequestFromState({ cloneJobRequest: { name: 'no-spec' } })).toBeNull();
  });
});
