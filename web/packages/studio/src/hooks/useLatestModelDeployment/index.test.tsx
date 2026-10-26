// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useModelsListDeploymentConfigs } from '@nemo/sdk/generated/platform/model-deployment-configs';
import { useModelsListDeployments } from '@nemo/sdk/generated/platform/model-deployments';
import type { ModelDeployment, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { useLatestModelDeployment } from '@studio/hooks/useLatestModelDeployment';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook } from '@testing-library/react';
import type { ReactNode } from 'react';

vi.mock('@nemo/sdk/generated/platform/model-deployment-configs');
vi.mock('@nemo/sdk/generated/platform/model-deployments');

const model = { name: 'my-model', workspace: 'ws' } as ModelEntity;

const mockDeployments = (deployments: Partial<ModelDeployment>[]) => {
  vi.mocked(useModelsListDeploymentConfigs).mockReturnValue({
    data: { data: [{ name: 'my-config' }] },
    isLoading: false,
  } as unknown as ReturnType<typeof useModelsListDeploymentConfigs>);
  vi.mocked(useModelsListDeployments).mockReturnValue({
    data: { data: deployments },
    isLoading: false,
  } as unknown as ReturnType<typeof useModelsListDeployments>);
};

const wrapper = ({ children }: { children: ReactNode }) => (
  <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
);

const renderLatest = () =>
  renderHook(() => useLatestModelDeployment(model, { isServed: false }), { wrapper }).result
    .current;

describe('useLatestModelDeployment', () => {
  it("looks up the model's deployments through the configs bound to it", () => {
    mockDeployments([]);
    renderLatest();

    expect(useModelsListDeploymentConfigs).toHaveBeenCalledWith(
      'ws',
      expect.objectContaining({ filter: { model_entity_id: 'ws/my-model' } }),
      expect.anything()
    );
    expect(useModelsListDeployments).toHaveBeenCalledWith(
      'ws',
      expect.objectContaining({ filter: { config: { $in: ['my-config'] } } }),
      expect.anything()
    );
  });

  it('returns the newest deployment that is not being deleted', () => {
    mockDeployments([
      { name: 'deleting', status: 'DELETING' },
      { name: 'starting', status: 'PENDING' },
      { name: 'older', status: 'ERROR' },
    ]);

    expect(renderLatest().deployment).toMatchObject({ name: 'starting' });
  });

  it('returns nothing when only deleted deployments remain', () => {
    mockDeployments([{ name: 'gone', status: 'DELETED' }]);

    expect(renderLatest().deployment).toBeUndefined();
  });
});
