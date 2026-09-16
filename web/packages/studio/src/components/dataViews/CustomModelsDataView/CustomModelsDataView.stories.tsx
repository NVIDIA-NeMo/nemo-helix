// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type ModelDeployment,
  ModelDeploymentStatus,
  type ModelEntity,
  type ModelEntitysPage,
  type ModelProvider,
} from '@nemo/sdk/generated/platform/schema';
import type { Meta, StoryObj } from '@storybook/react';
import { CustomModelsDataView } from '@studio/components/dataViews/CustomModelsDataView';
import {
  emptyModelEntitysPage,
  entityStoreCustomizedModel1,
  entityStorePromptTunedModel1,
} from '@studio/mocks/entity-store/models';
import { http, HttpResponse } from 'msw';

const MODELS_API = '/apis/models/v2/workspaces/:workspace/models';
const DEPLOYMENT_API = '/apis/models/v2/workspaces/:workspace/deployments/:name';

const moreCustomModels: ModelEntity[] = [
  {
    id: 'model-custom-2',
    name: 'llama-3.1-8b-instruct-sft-lora',
    workspace: 'default',
    created_at: '2025-02-10T09:15:32.123456',
    updated_at: '2025-02-10T10:45:12.654321',
    base_model: 'meta/llama-3.1-8b-instruct',
    adapters: [
      {
        name: 'lora-adapter',
        fileset: 'default/lora-fileset-2',
        finetuning_type: 'lora',
        workspace: 'default',
      },
    ],
    custom_fields: {},
  },
  {
    id: 'model-custom-3',
    name: 'qwen-2.5-72b-instruct-dpo-qlora',
    workspace: 'default',
    created_at: '2025-03-01T14:22:08.987654',
    updated_at: '2025-03-01T16:30:44.123456',
    base_model: 'qwen/qwen-2.5-72b-instruct',
    adapters: [
      {
        name: 'qlora-adapter',
        fileset: 'default/qlora-fileset',
        finetuning_type: 'qlora',
        workspace: 'default',
      },
    ],
    custom_fields: {},
  },
  {
    id: 'model-custom-4',
    name: 'mistral-7b-v0.3-grpo-all-weights',
    workspace: 'default',
    created_at: '2025-03-05T11:00:00.000000',
    updated_at: '2025-03-05T13:22:18.555555',
    base_model: 'mistralai/mistral-7b-instruct-v0.3',
    finetuning_type: 'all_weights',
    custom_fields: {},
  },
  {
    id: 'model-custom-5',
    name: 'nemotron-4-340b-distillation',
    workspace: 'default',
    created_at: '2025-04-12T08:30:00.000000',
    updated_at: '2025-04-12T12:15:33.111111',
    base_model: 'nvidia/nemotron-4-340b-instruct',
    finetuning_type: 'lora_merged',
    custom_fields: {},
  },
  {
    id: 'model-custom-6',
    name: 'llama-3.2-1b-sft-prompt-tuned',
    workspace: 'default',
    created_at: '2025-05-20T17:45:00.000000',
    updated_at: '2025-05-20T18:10:22.333333',
    base_model: 'meta/llama-3.2-1b-instruct',
    prompt: {
      system_prompt: 'You are a helpful coding assistant.',
    },
    custom_fields: {},
  },
  {
    id: 'model-custom-7',
    name: 'codellama-70b-dataset-XyZ123-dora',
    workspace: 'default',
    created_at: '2025-06-01T22:00:00.000000',
    updated_at: '2025-06-02T03:45:10.777777',
    base_model: 'default/codellama-70b',
    adapters: [
      {
        name: 'dora-adapter',
        fileset: 'default/dora-fileset',
        finetuning_type: 'dora',
        workspace: 'default',
      },
    ],
    custom_fields: {},
  },
  {
    id: 'model-custom-8',
    name: 'gemma-2-9b-it-lora-customer-support',
    workspace: 'default',
    created_at: '2025-06-15T10:20:00.000000',
    updated_at: '2025-06-15T11:55:44.222222',
    base_model: 'google/gemma-2-9b-it',
    adapters: [
      {
        name: 'lora-adapter',
        fileset: 'default/lora-fileset-3',
        finetuning_type: 'lora',
        workspace: 'default',
      },
    ],
    custom_fields: {},
  },
];

const customModelsPage: ModelEntitysPage = {
  data: [entityStorePromptTunedModel1, entityStoreCustomizedModel1, ...moreCustomModels],
  pagination: {
    page: 1,
    page_size: 10,
    current_page_size: 9,
    total_pages: 1,
    total_results: 9,
  },
};

const makeDeployment = (name: string, status: ModelDeploymentStatus): ModelDeployment => ({
  name,
  workspace: 'default',
  created_at: '2025-06-01T00:00:00.000000',
  updated_at: '2025-06-01T00:00:00.000000',
  entity_version: 1,
  config: `${name}-config`,
  config_version: 1,
  status,
});

const deploymentsByName: Record<string, ModelDeployment> = {
  'some-custom-model': makeDeployment('some-custom-model', ModelDeploymentStatus.READY),
  'codellama-70b-dataset-AnxDxZ6MBzFprTU78BAYP9-lora': makeDeployment(
    'codellama-70b-dataset-AnxDxZ6MBzFprTU78BAYP9-lora',
    ModelDeploymentStatus.READY
  ),
  'llama-3.1-8b-instruct-sft-lora': makeDeployment(
    'llama-3.1-8b-instruct-sft-lora',
    ModelDeploymentStatus.PENDING
  ),
  'qwen-2.5-72b-instruct-dpo-qlora': makeDeployment(
    'qwen-2.5-72b-instruct-dpo-qlora',
    ModelDeploymentStatus.ERROR
  ),
  'mistral-7b-v0.3-grpo-all-weights': makeDeployment(
    'mistral-7b-v0.3-grpo-all-weights',
    ModelDeploymentStatus.READY
  ),
  'nemotron-4-340b-distillation': makeDeployment(
    'nemotron-4-340b-distillation',
    ModelDeploymentStatus.CREATED
  ),
  'codellama-70b-dataset-XyZ123-dora': makeDeployment(
    'codellama-70b-dataset-XyZ123-dora',
    ModelDeploymentStatus.READY
  ),
  'gemma-2-9b-it-lora-customer-support': makeDeployment(
    'gemma-2-9b-it-lora-customer-support',
    ModelDeploymentStatus.DELETING
  ),
};

const meta = {
  component: CustomModelsDataView,
  title: 'DataViews/CustomModelsDataView',
  args: {
    workspace: 'default',
  },
} satisfies Meta<typeof CustomModelsDataView>;

export default meta;
type Story = StoryObj<typeof meta>;

const deploymentHandler = http.get<{ name: string }>(DEPLOYMENT_API, ({ params }) => {
  const deployment = deploymentsByName[params.name];
  if (!deployment) return new HttpResponse(null, { status: 404 });
  return HttpResponse.json(deployment);
});

export const Empty: Story = {
  parameters: {
    msw: {
      handlers: [
        http.get<never, never, ModelEntitysPage>(MODELS_API, () =>
          HttpResponse.json(emptyModelEntitysPage)
        ),
      ],
    },
  },
};

export const WithData: Story = {
  parameters: {
    msw: {
      handlers: [
        http.get<never, never, ModelEntitysPage>(MODELS_API, () =>
          HttpResponse.json(customModelsPage)
        ),
        deploymentHandler,
      ],
    },
  },
};

/* -------------------------------------------------------------------------- */
/* Status column                                                              */
/* -------------------------------------------------------------------------- */

/**
 * One row per Status badge. Several of these states cannot be produced by real
 * data on demand -- Not served needs a base redeployed with LoRA disabled, and
 * Unknown needs a provider request to fail -- so this story is the only place they
 * can be reviewed side by side.
 *
 * Status resolves through models -> `model_providers` -> provider `served_models`
 * -> deployment, so each row below needs a provider mock as well as a deployment.
 */
const PROVIDER_API = '/apis/models/v2/workspaces/:workspace/providers/:name';

const statusModel = (name: string, overrides: Partial<ModelEntity> = {}): ModelEntity =>
  ({
    id: `model-${name}`,
    name,
    workspace: 'default',
    created_at: '2025-06-01T00:00:00.000000',
    updated_at: '2025-06-01T00:00:00.000000',
    base_model: 'meta/llama-3.1-8b-instruct',
    finetuning_type: 'all_weights',
    model_providers: [`default/provider-${name}`],
    ...overrides,
  }) as ModelEntity;

const LORA_HOST = 'lora-host-model';

const statusModels: ModelEntity[] = [
  statusModel('ready-deployment'),
  // Parent reads Deployed; its two adapters read Served and Not served.
  statusModel(LORA_HOST, {
    base_model: undefined,
    finetuning_type: undefined,
    adapters: [
      {
        name: 'adapter-loaded',
        workspace: 'default',
        fileset: 'default/adapter-loaded-fileset',
        finetuning_type: 'lora',
      },
      {
        name: 'adapter-not-loaded',
        workspace: 'default',
        fileset: 'default/adapter-not-loaded-fileset',
        finetuning_type: 'lora',
      },
    ],
  }),
  statusModel('pending-deployment'),
  statusModel('created-deployment'),
  statusModel('error-deployment'),
  statusModel('deleting-deployment'),
  statusModel('lost-deployment'),
  // Served by a provider that names no deployment, e.g. build.nvidia.com.
  statusModel('external-provider'),
  // Its provider request fails, so whether it is served is genuinely unknown.
  statusModel('unreadable-provider'),
  // No providers at all: nothing serves it, and we know that for certain.
  statusModel('no-providers', { model_providers: [] }),
];

const statusModelsPage: ModelEntitysPage = {
  data: statusModels,
  pagination: {
    page: 1,
    page_size: 10,
    current_page_size: statusModels.length,
    total_pages: 1,
    total_results: statusModels.length,
  },
};

const makeProvider = (
  name: string,
  servedEntityIds: string[],
  deploymentName: string | null
): ModelProvider =>
  ({
    id: `provider-${name}`,
    name: `provider-${name}`,
    workspace: 'default',
    created_at: '2025-06-01T00:00:00.000000',
    updated_at: '2025-06-01T00:00:00.000000',
    host_url: 'http://localhost:8000',
    status: 'READY',
    model_deployment_id: deploymentName ? `default/${deploymentName}` : undefined,
    served_models: servedEntityIds.map((id) => ({
      model_entity_id: id,
      served_model_name: id.replace(/\//g, '--'),
    })),
  }) as ModelProvider;

const statusProvidersByName: Record<string, ModelProvider> = {
  'provider-ready-deployment': makeProvider(
    'ready-deployment',
    ['default/ready-deployment'],
    'dep-ready'
  ),
  [`provider-${LORA_HOST}`]: makeProvider(
    LORA_HOST,
    [
      `default/${LORA_HOST}`,
      // Only the loaded adapter is registered by the running backend.
      `default/${LORA_HOST}&adapters/default/adapter-loaded`,
    ],
    'dep-lora'
  ),
  'provider-pending-deployment': makeProvider(
    'pending-deployment',
    ['default/pending-deployment'],
    'dep-pending'
  ),
  'provider-created-deployment': makeProvider(
    'created-deployment',
    ['default/created-deployment'],
    'dep-created'
  ),
  'provider-error-deployment': makeProvider(
    'error-deployment',
    ['default/error-deployment'],
    'dep-error'
  ),
  'provider-deleting-deployment': makeProvider(
    'deleting-deployment',
    ['default/deleting-deployment'],
    'dep-deleting'
  ),
  'provider-lost-deployment': makeProvider(
    'lost-deployment',
    ['default/lost-deployment'],
    'dep-lost'
  ),
  'provider-external-provider': makeProvider(
    'external-provider',
    ['default/external-provider'],
    null
  ),
};

const statusDeploymentsByName: Record<string, ModelDeployment> = {
  'dep-ready': makeDeployment('dep-ready', ModelDeploymentStatus.READY),
  'dep-lora': makeDeployment('dep-lora', ModelDeploymentStatus.READY),
  'dep-pending': makeDeployment('dep-pending', ModelDeploymentStatus.PENDING),
  'dep-created': makeDeployment('dep-created', ModelDeploymentStatus.CREATED),
  'dep-error': makeDeployment('dep-error', ModelDeploymentStatus.ERROR),
  'dep-deleting': makeDeployment('dep-deleting', ModelDeploymentStatus.DELETING),
  'dep-lost': makeDeployment('dep-lost', ModelDeploymentStatus.LOST),
};

/**
 * Expected badges, top to bottom (expand the LoRA host row for its adapters):
 *
 * | Row                   | Status       |
 * | --------------------- | ------------ |
 * | ready-deployment      | Deployed     |
 * | lora-host-model       | Deployed     |
 * |   adapter-loaded      | Served       |
 * |   adapter-not-loaded  | Not served   |
 * | pending-deployment    | Deploying    |
 * | created-deployment    | Deploying    |
 * | error-deployment      | Failed       |
 * | deleting-deployment   | Deleting     |
 * | lost-deployment       | Unavailable  |
 * | external-provider     | Available    |
 * | unreadable-provider   | Unknown      |
 * | no-providers          | Not deployed |
 */
export const AllDeploymentStatuses: Story = {
  parameters: {
    msw: {
      handlers: [
        http.get<never, never, ModelEntitysPage>(MODELS_API, () =>
          HttpResponse.json(statusModelsPage)
        ),
        http.get<{ name: string }>(PROVIDER_API, ({ params }) => {
          const provider = statusProvidersByName[params.name];
          // `provider-unreadable-provider` is deliberately absent: the request
          // fails, which must read as Unknown rather than Not deployed.
          if (!provider) return new HttpResponse(null, { status: 500 });
          return HttpResponse.json(provider);
        }),
        http.get<{ name: string }>(DEPLOYMENT_API, ({ params }) => {
          const deployment = statusDeploymentsByName[params.name];
          if (!deployment) return new HttpResponse(null, { status: 404 });
          return HttpResponse.json(deployment);
        }),
      ],
    },
  },
};
