// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Engine, type ModelDeploymentConfig } from '@nemo/sdk/generated/platform/schema';
import {
  getConfigFields,
  getConfigTiles,
} from '@studio/routes/ModelDetailRoute/DeploymentTab/configFields';

const config = {
  name: 'my-config',
  workspace: 'ws',
  created_at: '2026-10-22T00:00:00Z',
  updated_at: '2026-10-22T00:00:00Z',
  entity_version: 1,
  engine: Engine.vllm,
  model_spec: {
    model_namespace: 'ws',
    model_name: 'my-model',
    model_revision: 'main',
    lora_enabled: true,
  },
  executor_config: {
    gpu: 2,
    image_name: 'vllm/vllm-openai',
    image_tag: 'v0.11.0',
    disk_size: '100Gi',
    additional_args: ['--max-model-len', '32768'],
    additional_envs: { HF_TOKEN: 'secret-token', LOG_LEVEL: 'debug' },
  },
} as ModelDeploymentConfig;

const valueOf = (label: string, cfg = config) =>
  getConfigFields(cfg).find((field) => field.label === label)?.value;

describe('getConfigTiles', () => {
  it('shows the engine, GPUs, disk and LoRA', () => {
    expect(getConfigTiles(config)).toEqual([
      { label: 'Engine', value: 'vLLM' },
      { label: 'GPUs', value: '2' },
      { label: 'Disk', value: '100Gi' },
      { label: 'LoRA', value: 'Enabled' },
    ]);
  });
});

describe('getConfigFields', () => {
  it('formats the source and image', () => {
    expect(valueOf('Source')).toBe('ws/my-model @ main');
    expect(valueOf('Image')).toBe('vllm/vllm-openai:v0.11.0');
    expect(valueOf('Arguments')).toBe('--max-model-len 32768');
  });

  it('lists environment variable names without their values', () => {
    expect(valueOf('Environment')).toBe('HF_TOKEN, LOG_LEVEL');
    const rendered = JSON.stringify(getConfigFields(config));
    expect(rendered).not.toContain('secret-token');
  });

  it('says an override is set without repeating it', () => {
    expect(valueOf('Chat Template')).toBeUndefined();
    expect(
      valueOf('Chat Template', {
        ...config,
        model_spec: { ...config.model_spec, chat_template: '{{ messages }}' },
      })
    ).toBe('Overridden');
  });
});
