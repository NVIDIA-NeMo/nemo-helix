// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ModelDeploymentStatus } from '@nemo/sdk/generated/platform/schema';
import { useBaseModelDeploymentReadiness } from '@studio/hooks/useBaseModelDeploymentReadiness';
import { renderHook } from '@testing-library/react';

const mocks = vi.hoisted(() => ({
  useModelsGetModel: vi.fn(),
  useModelsGetProvider: vi.fn(),
  useModelsGetLatestDeployment: vi.fn(),
  useModelsGetDeploymentConfigVersion: vi.fn(),
}));

vi.mock('@nemo/sdk/generated/platform/models', () => ({
  useModelsGetModel: mocks.useModelsGetModel,
}));
vi.mock('@nemo/sdk/generated/platform/model-providers', () => ({
  useModelsGetProvider: mocks.useModelsGetProvider,
}));
vi.mock('@nemo/sdk/generated/platform/model-deployments', () => ({
  useModelsGetLatestDeployment: mocks.useModelsGetLatestDeployment,
}));
vi.mock('@nemo/sdk/generated/platform/model-deployment-configs', () => ({
  useModelsGetDeploymentConfigVersion: mocks.useModelsGetDeploymentConfigVersion,
}));

const idle = { data: undefined, isLoading: false };

function setup(opts: {
  providers?: string[];
  deploymentId?: string;
  status?: ModelDeploymentStatus;
  loraEnabled?: boolean;
}) {
  mocks.useModelsGetModel.mockReturnValue({
    data: { workspace: 'ws', name: 'base', model_providers: opts.providers },
    isLoading: false,
  });
  mocks.useModelsGetProvider.mockReturnValue(
    opts.deploymentId
      ? { data: { model_deployment_id: opts.deploymentId }, isLoading: false }
      : idle
  );
  mocks.useModelsGetLatestDeployment.mockReturnValue(
    opts.status
      ? {
          data: {
            name: 'base-deployment',
            status: opts.status,
            config: 'base-config',
            config_version: 1,
          },
          isLoading: false,
        }
      : idle
  );
  mocks.useModelsGetDeploymentConfigVersion.mockReturnValue(
    opts.loraEnabled === undefined
      ? idle
      : { data: { model_spec: { lora_enabled: opts.loraEnabled } }, isLoading: false }
  );
}

const servingBase = {
  providers: ['ws/p'],
  deploymentId: 'ws/base-deployment',
  status: ModelDeploymentStatus.READY,
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe('useBaseModelDeploymentReadiness', () => {
  it('reports none when the model has no providers', () => {
    setup({});
    expect(renderHook(() => useBaseModelDeploymentReadiness('ws/base')).result.current.state).toBe(
      'none'
    );
  });

  it('reports serving-lora for a READY deployment with lora_enabled', () => {
    setup({ ...servingBase, loraEnabled: true });
    const { result } = renderHook(() => useBaseModelDeploymentReadiness('ws/base'));
    expect(result.current.state).toBe('serving-lora');
    expect(result.current.deploymentName).toBe('base-deployment');
  });

  // The silent failure this whole flow exists to prevent: the base is up, so
  // nothing looks wrong, but it will refuse to serve any adapter trained on it.
  it('reports serving-without-lora when lora_enabled is false', () => {
    setup({ ...servingBase, loraEnabled: false });
    expect(renderHook(() => useBaseModelDeploymentReadiness('ws/base')).result.current.state).toBe(
      'serving-without-lora'
    );
  });

  it('treats PENDING as live, not missing', () => {
    setup({ ...servingBase, status: ModelDeploymentStatus.PENDING, loraEnabled: true });
    expect(renderHook(() => useBaseModelDeploymentReadiness('ws/base')).result.current.state).toBe(
      'serving-lora'
    );
  });

  it('reports unavailable for a terminal deployment', () => {
    setup({ ...servingBase, status: ModelDeploymentStatus.ERROR, loraEnabled: true });
    expect(renderHook(() => useBaseModelDeploymentReadiness('ws/base')).result.current.state).toBe(
      'unavailable'
    );
  });

  // Reads the version the deployment pinned, not the latest: configs are
  // immutable and versioned, so the latest may describe something not running.
  it('reads the config version the deployment pinned', () => {
    setup({ ...servingBase, loraEnabled: true });
    renderHook(() => useBaseModelDeploymentReadiness('ws/base'));
    expect(mocks.useModelsGetDeploymentConfigVersion).toHaveBeenCalledWith(
      'ws',
      'base-config',
      '1',
      expect.objectContaining({ query: expect.objectContaining({ enabled: true }) })
    );
  });

  it('runs no queries without a model ref', () => {
    setup({});
    renderHook(() => useBaseModelDeploymentReadiness(undefined));
    expect(mocks.useModelsGetModel.mock.calls[0][3]).toMatchObject({
      query: { enabled: false },
    });
  });

  it('runs no queries when explicitly disabled', () => {
    setup({});
    renderHook(() => useBaseModelDeploymentReadiness('ws/base', { enabled: false }));
    expect(mocks.useModelsGetModel.mock.calls[0][3]).toMatchObject({
      query: { enabled: false },
    });
  });
});
