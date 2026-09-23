// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  buildSeedPath,
  buildTransformJobRequest,
} from '@studio/components/DataDesignerTransformModal/buildTransformJobRequest';

const input = {
  jobName: 'support-evals-agent-eval-tasks',
  processorName: 'agent_eval_tasks',
  filesetWorkspace: 'default',
  filesetName: 'support-evals-artifacts',
  filePath: 'dataset.parquet',
  numRecords: 500,
  template: { id: '{{ task_id }}' },
};

describe('buildSeedPath', () => {
  it('addresses a single file inside a workspace-qualified fileset', () => {
    expect(buildSeedPath('default', 'my-fileset', 'nested/dataset.parquet')).toBe(
      'default/my-fileset#nested/dataset.parquet'
    );
  });
});

describe('buildTransformJobRequest', () => {
  it('declares no columns so the job generates nothing', () => {
    expect(buildTransformJobRequest(input).spec.config.columns).toEqual([]);
  });

  it('declares a UUID sampler when an id column has to be generated', () => {
    const request = buildTransformJobRequest({ ...input, generatedIdColumn: 'row_id' });
    expect(request.spec.config.columns).toEqual([
      {
        name: 'row_id',
        column_type: 'sampler',
        sampler_type: 'uuid',
        params: { short_form: true },
      },
    ]);
    // A sampler needs no model, so the job is still inference-free.
    expect(request.spec.config.model_configs).toBeUndefined();
  });

  it('seeds from the source file in order', () => {
    const { seed_config: seedConfig } = buildTransformJobRequest(input).spec.config;
    expect(seedConfig).toEqual({
      source: { seed_type: 'nhx', path: 'default/support-evals-artifacts#dataset.parquet' },
      sampling_strategy: 'ordered',
    });
  });

  it('carries the template through on a schema_transform processor', () => {
    expect(buildTransformJobRequest(input).spec.config.processors).toEqual([
      {
        processor_type: 'schema_transform',
        name: 'agent_eval_tasks',
        template: { id: '{{ task_id }}' },
      },
    ]);
  });

  it('requests no models, since nothing is generated', () => {
    expect(buildTransformJobRequest(input).spec.config.model_configs).toBeUndefined();
  });

  it('passes the row count and job name through', () => {
    const request = buildTransformJobRequest(input);
    expect(request.name).toBe('support-evals-agent-eval-tasks');
    expect(request.spec.num_records).toBe(500);
  });
});
