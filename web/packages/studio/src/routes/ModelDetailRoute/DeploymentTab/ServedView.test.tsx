// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useProviderReady } from '@nemo/sdk/generated/platform/inference-gateway';
import { useModelsGetDeploymentConfigVersion } from '@nemo/sdk/generated/platform/model-deployment-configs';
import { useModelsGetLatestDeployment } from '@nemo/sdk/generated/platform/model-deployments';
import { useModelsGetProvider } from '@nemo/sdk/generated/platform/model-providers';
import {
  type Adapter,
  Engine,
  type ModelDeployment,
  type ModelDeploymentConfig,
  ModelDeploymentStatus,
  type ModelEntity,
  type ModelProvider,
} from '@nemo/sdk/generated/platform/schema';
import { useGetVirtualModel } from '@nemo/sdk/generated/platform/virtual-models';
import { useDeleteDeploymentAndConfig } from '@studio/routes/DeploymentsListRoute/useDeleteDeploymentAndConfig';
import { ServedView } from '@studio/routes/ModelDetailRoute/DeploymentTab/ServedView';
import { render, screen } from '@studio/tests/util/render';

vi.mock('@nemo/sdk/generated/platform/inference-gateway');
vi.mock('@nemo/sdk/generated/platform/model-deployment-configs');
vi.mock('@nemo/sdk/generated/platform/model-deployments');
vi.mock('@nemo/sdk/generated/platform/model-providers');
vi.mock('@nemo/sdk/generated/platform/virtual-models');
vi.mock('@studio/routes/DeploymentsListRoute/useDeleteDeploymentAndConfig');

const loadedAdapter = { name: 'loaded-lora', workspace: 'ws' } as Adapter;
const missingAdapter = { name: 'missing-lora', workspace: 'ws' } as Adapter;
const model = {
  name: 'my-model',
  workspace: 'ws',
  adapters: [loadedAdapter, missingAdapter],
} as ModelEntity;

const query = <T,>(data: T | undefined, extra: Record<string, unknown> = {}) =>
  ({ data, isLoading: false, isSuccess: data !== undefined, ...extra }) as never;

describe('ServedView', () => {
  beforeEach(() => {
    vi.mocked(useDeleteDeploymentAndConfig).mockReturnValue({
      deleteDeploymentAndConfig: vi.fn(),
      isDeleting: false,
    });
    vi.mocked(useModelsGetLatestDeployment).mockReturnValue(
      query({
        name: 'my-deployment',
        workspace: 'ws',
        status: ModelDeploymentStatus.READY,
        config: 'my-config',
        config_version: 1,
      } as ModelDeployment)
    );
    vi.mocked(useModelsGetDeploymentConfigVersion).mockReturnValue(
      query({
        engine: Engine.vllm,
        model_spec: { lora_enabled: true },
        executor_config: { gpu: 2 },
      } as ModelDeploymentConfig)
    );
    vi.mocked(useModelsGetProvider).mockReturnValue(
      query({
        name: 'my-provider',
        host_url: 'http://10.0.0.1:8000',
        served_models: [
          { model_entity_id: 'ws/my-model', served_model_name: 'my-model' },
          {
            model_entity_id: 'ws/my-model&adapters/ws/loaded-lora',
            served_model_name: 'ws--loaded-lora',
          },
        ],
      } as ModelProvider)
    );
    vi.mocked(useProviderReady).mockReturnValue(query({ ready: true }));
    vi.mocked(useGetVirtualModel).mockReturnValue(
      query({ name: 'my-model', workspace: 'ws', guardrail_config_ids: ['content-safety'] })
    );
  });

  it('shows how the deployment serves the model', () => {
    render(
      <ServedView
        model={model}
        adapter={null}
        deploymentRef="ws/my-deployment"
        providerRef="ws/my-provider"
      />
    );

    expect(screen.getByRole('link', { name: /my-deployment/ })).toBeInTheDocument();
    expect(screen.getByText('Routable')).toBeInTheDocument();
    expect(screen.getByText('http://10.0.0.1:8000')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Configuration' })).toBeInTheDocument();
    expect(screen.getByText('vLLM')).toBeInTheDocument();
    expect(screen.getByText('content-safety')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /undeploy/i })).not.toBeInTheDocument();
  });

  it('marks which adapters the deployment has loaded, leading with the current one', () => {
    render(
      <ServedView
        model={model}
        adapter={missingAdapter}
        deploymentRef="ws/my-deployment"
        providerRef="ws/my-provider"
      />
    );

    expect(screen.getByText('1 of 2 loaded on this deployment.')).toBeInTheDocument();
    const cards = screen.getAllByRole('link', { name: /-lora/ });
    expect(cards[0]).toHaveTextContent('missing-lora');
    expect(cards[0]).toHaveTextContent('Not active');
    expect(cards[1]).toHaveTextContent('loaded-lora');
    expect(cards[1]).toHaveTextContent('Active');
  });

  it('marks an external provider, with no deployment to show', () => {
    vi.mocked(useModelsGetLatestDeployment).mockReturnValue(query(undefined));
    render(<ServedView model={model} adapter={null} providerRef="default/build" />);

    expect(screen.getByText('External provider')).toBeInTheDocument();
    expect(screen.queryByText('Deployment')).not.toBeInTheDocument();
  });
});
