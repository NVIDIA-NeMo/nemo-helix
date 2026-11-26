// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  PII_MODEL_ID_REQUIRED,
  PII_MODEL_PROVIDER_REQUIRED,
  type SafeSynthesizerFormData,
  getSafeSynthesizerFormDefaults,
  safeSynthesizerJobRequestSchema,
  toGenerateJobRequest,
} from '@studio/routes/SafeSynthesizerNewRoute/schema';

const formData = ({
  enableReplacePii = true,
  provider = '',
  modelId = '',
}: {
  enableReplacePii?: boolean;
  provider?: string;
  modelId?: string;
}): SafeSynthesizerFormData => {
  const defaults = getSafeSynthesizerFormDefaults();
  return {
    ...defaults,
    spec: {
      ...defaults.spec,
      data_source: 'test-workspace/fileset#data.csv',
      pii_model_provider: provider,
      config: {
        ...defaults.spec.config,
        enable_replace_pii: enableReplacePii,
        replace_pii: { llm: { model_id: modelId } },
      },
    },
  };
};

const issueMessages = (data: SafeSynthesizerFormData) => {
  const result = safeSynthesizerJobRequestSchema.safeParse(data);
  if (result.success) {
    return [];
  }
  return result.error.issues.map((issue) => `${issue.path.join('.')}: ${issue.message}`);
};

describe('safeSynthesizerJobRequestSchema PII discovery model', () => {
  it('requires a provider and a model ID while PII replacement is on', () => {
    expect(issueMessages(formData({}))).toEqual([
      `spec.pii_model_provider: ${PII_MODEL_PROVIDER_REQUIRED}`,
      `spec.config.replace_pii.llm.model_id: ${PII_MODEL_ID_REQUIRED}`,
    ]);
  });

  it('requires a model ID when only the provider is set', () => {
    expect(
      issueMessages(formData({ provider: 'test-workspace/pii-provider', modelId: '  ' }))
    ).toEqual([`spec.config.replace_pii.llm.model_id: ${PII_MODEL_ID_REQUIRED}`]);
  });

  it('requires a provider when only the model ID is set', () => {
    expect(issueMessages(formData({ modelId: 'nvidia/nemotron' }))).toEqual([
      `spec.pii_model_provider: ${PII_MODEL_PROVIDER_REQUIRED}`,
    ]);
  });

  it('accepts a provider and model ID', () => {
    expect(
      issueMessages(
        formData({ provider: 'test-workspace/pii-provider', modelId: 'nvidia/nemotron' })
      )
    ).toEqual([]);
  });

  it('requires neither while PII replacement is off', () => {
    expect(issueMessages(formData({ enableReplacePii: false }))).toEqual([]);
  });
});

describe('toGenerateJobRequest', () => {
  it('sends the PII provider and trimmed model ID while PII replacement is on', () => {
    const request = toGenerateJobRequest(
      formData({ provider: 'test-workspace/pii-provider', modelId: ' nvidia/nemotron ' })
    );

    expect(request.spec.pii_model_provider).toBe('test-workspace/pii-provider');
    expect(request.spec.config.enable_replace_pii).toBe(true);
    expect(request.spec.config.replace_pii).toEqual({ llm: { model_id: 'nvidia/nemotron' } });
  });

  it('sends only enable_replace_pii: false when PII replacement is off', () => {
    const request = toGenerateJobRequest(
      formData({
        enableReplacePii: false,
        provider: 'test-workspace/pii-provider',
        modelId: 'nvidia/nemotron',
      })
    );

    expect(request.spec).not.toHaveProperty('pii_model_provider');
    expect(request.spec.config).not.toHaveProperty('replace_pii');
    expect(request.spec.config.enable_replace_pii).toBe(false);
  });

  it('drops blank example grouping and ordering without touching the form values', () => {
    const data = formData({ enableReplacePii: false });
    data.spec.config.data = {
      group_training_examples_by: 'user_id',
      order_training_examples_by: ' ',
    };

    const request = toGenerateJobRequest(data);

    expect(request.spec.config.data).toEqual({ group_training_examples_by: 'user_id' });
    expect(data.spec.config.data.order_training_examples_by).toBe(' ');
  });
});
