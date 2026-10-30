// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  MODEL_SIZE_OPTIONS,
  type ModelFilterValues,
} from '@nemo/common/src/api/models/modelFilters';
import type { ModelWorkspaceGroup } from '@nemo/common/src/api/models/useModels';
import { ModelSelectV2 } from '@nemo/common/src/components/ModelSelectV2/ModelSelectV2';
import type {
  ModelSelection,
  ModelSelectV2Props,
} from '@nemo/common/src/components/ModelSelectV2/types';
import type { InferenceParams } from '@nemo/sdk/generated/platform/schema';
import type { Meta, StoryObj } from '@storybook/react';
import { useState } from 'react';

const mockGroups: ModelWorkspaceGroup[] = [
  {
    workspace: 'nvidia',
    models: [
      {
        id: '1',
        name: 'llama-3.1-8b-instruct',
        workspace: 'nvidia',
        created_at: '2025-12-01T00:00:00Z',
        updated_at: '2025-12-01T00:00:00Z',
        description: 'Llama 3.1 8B instruction-tuned model',
        adapters: [
          {
            name: 'customer-support-v2',
            fileset: 'nvidia/cs-v2',
            finetuning_type: 'lora' as const,
            created_at: '2026-02-15T00:00:00Z',
            workspace: 'nvidia',
          },
          {
            name: 'customer-support-v1',
            fileset: 'nvidia/cs-v1',
            finetuning_type: 'lora' as const,
            created_at: '2026-01-10T00:00:00Z',
            workspace: 'nvidia',
          },
        ],
        finetuning_type: 'lora',
        base_model: 'nvidia/llama-3.1-8b',
      },
      {
        id: '2',
        name: 'nemotron-8b',
        workspace: 'nvidia',
        created_at: '2025-11-01T00:00:00Z',
        updated_at: '2025-11-01T00:00:00Z',
        description: 'Nemotron 8B base model for general tasks',
      },
      {
        id: '3',
        name: 'nemotron-70b',
        workspace: 'nvidia',
        created_at: '2025-10-01T00:00:00Z',
        updated_at: '2025-10-01T00:00:00Z',
        description: 'Nemotron 70B large model',
      },
    ],
  },
  {
    workspace: 'meta',
    models: [
      {
        id: '4',
        name: 'llama-3.1-70b@v1.2',
        workspace: 'meta',
        created_at: '2025-09-01T00:00:00Z',
        updated_at: '2025-09-01T00:00:00Z',
        description: 'Llama 3.1 70B base model',
      },
      {
        id: '5',
        name: 'llama-3.1-8b',
        workspace: 'meta',
        created_at: '2025-08-01T00:00:00Z',
        updated_at: '2025-08-01T00:00:00Z',
        description: 'Llama 3.1 8B base model',
      },
    ],
  },
  {
    workspace: 'mistral',
    models: [
      {
        id: '6',
        name: 'mistral-7b-instruct',
        workspace: 'mistral',
        created_at: '2025-07-01T00:00:00Z',
        updated_at: '2025-07-01T00:00:00Z',
        description: 'Mistral 7B instruction-tuned model',
        finetuning_type: 'lora',
        base_model: 'mistral/mistral-7b',
      },
    ],
  },
];

const groupByWorkspace = (models: ModelWorkspaceGroup['models']): ModelWorkspaceGroup[] => {
  const byWorkspace = new Map<string, ModelWorkspaceGroup['models']>();
  for (const model of models) {
    const workspace = model.workspace ?? 'default';
    byWorkspace.set(workspace, [...(byWorkspace.get(workspace) ?? []), model]);
  }
  return Array.from(byWorkspace, ([workspace, groupModels]) => ({
    workspace,
    models: groupModels,
  }));
};

const meta: Meta<typeof ModelSelectV2> = {
  component: ModelSelectV2,
  title: 'Studio Common/ModelSelectV2',
};

export default meta;

type Story = StoryObj<typeof ModelSelectV2>;

const DefaultRender = (args: ModelSelectV2Props) => {
  const [value, setValue] = useState<ModelSelection | null>(null);
  return <ModelSelectV2 {...args} value={value} onValueChange={setValue} groups={mockGroups} />;
};

export const Default: Story = {
  render: DefaultRender,
};

const WithSelectionRender = (args: ModelSelectV2Props) => {
  const [value, setValue] = useState<ModelSelection | null>({
    model: 'nvidia/nemotron-8b',
  });
  return <ModelSelectV2 {...args} value={value} onValueChange={setValue} groups={mockGroups} />;
};

export const WithSelection: Story = {
  render: WithSelectionRender,
};

const WithModelTypeToggleRender = (args: ModelSelectV2Props) => {
  const [value, setValue] = useState<ModelSelection | null>(null);
  return (
    <ModelSelectV2
      {...args}
      value={value}
      onValueChange={setValue}
      groups={mockGroups}
      showModelTypeToggle
    />
  );
};

export const WithModelTypeToggle: Story = {
  render: WithModelTypeToggleRender,
};

const MOCK_FAMILIES = ['llama', 'mistral', 'nemotron'];

const WithFiltersRender = (args: ModelSelectV2Props) => {
  const [value, setValue] = useState<ModelSelection | null>(null);
  const [filters, setFilters] = useState<ModelFilterValues>({});
  const groups = mockGroups
    .filter((group) => !filters.provider || group.workspace === filters.provider)
    .map((group) => ({
      ...group,
      models: group.models.filter(
        (model) => !filters.family || model.name.startsWith(filters.family)
      ),
    }))
    .filter((group) => group.models.length > 0);

  return (
    <ModelSelectV2
      {...args}
      value={value}
      onValueChange={setValue}
      groups={groups}
      modelFilters={{
        values: filters,
        onChange: setFilters,
        providerOptions: mockGroups.map(({ workspace }) => ({
          value: workspace,
          label: workspace,
        })),
        familyOptions: MOCK_FAMILIES.map((family) => ({ value: family, label: family })),
        sizeOptions: MODEL_SIZE_OPTIONS,
        providersLoading: false,
      }}
    />
  );
};

export const WithFilters: Story = {
  render: WithFiltersRender,
};

const WithParamsRender = (args: ModelSelectV2Props) => {
  const [value, setValue] = useState<ModelSelection | null>(null);
  const [inferenceParams, setInferenceParams] = useState<Partial<InferenceParams>>({});
  return (
    <ModelSelectV2
      {...args}
      value={value}
      onValueChange={setValue}
      groups={mockGroups}
      showParams
      inferenceParams={inferenceParams}
      onInferenceParamsChange={setInferenceParams}
    />
  );
};

export const WithParams: Story = {
  render: WithParamsRender,
};

const WithAllFeaturesRender = (args: ModelSelectV2Props) => {
  const [value, setValue] = useState<ModelSelection | null>(null);
  const [inferenceParams, setInferenceParams] = useState<Partial<InferenceParams>>({});
  return (
    <ModelSelectV2
      {...args}
      fullWidth
      value={value}
      onValueChange={setValue}
      groups={mockGroups}
      showModelTypeToggle
      showParams
      inferenceParams={inferenceParams}
      onInferenceParamsChange={setInferenceParams}
    />
  );
};

export const WithAllFeatures: Story = {
  render: WithAllFeaturesRender,
};

const LoadingRender = (args: ModelSelectV2Props) => {
  const [value, setValue] = useState<ModelSelection | null>(null);
  return <ModelSelectV2 {...args} value={value} onValueChange={setValue} groups={[]} loading />;
};

export const Loading: Story = {
  render: LoadingRender,
};

const DisabledRender = (args: ModelSelectV2Props) => {
  const [value, setValue] = useState<ModelSelection | null>(null);
  return (
    <ModelSelectV2 {...args} value={value} onValueChange={setValue} groups={mockGroups} disabled />
  );
};

export const Disabled: Story = {
  render: DisabledRender,
};

const EmptyRender = (args: ModelSelectV2Props) => {
  const [value, setValue] = useState<ModelSelection | null>(null);
  return <ModelSelectV2 {...args} value={value} onValueChange={setValue} groups={[]} />;
};

export const Empty: Story = {
  render: EmptyRender,
};

const manyModelsGroups: ModelWorkspaceGroup[] = [
  {
    workspace: 'nvidia',
    models: [
      'llama-3.1-8b-instruct',
      'llama-3.1-70b-instruct',
      'llama-3.3-nemotron-super-49b',
      'nemotron-8b',
      'nemotron-70b',
      'nemotron-340b',
      'minitron-4b',
      'minitron-8b',
    ].map((name, i) => ({
      id: `nv-${i}`,
      name,
      workspace: 'nvidia',
      created_at: '2025-12-01T00:00:00Z',
      updated_at: '2025-12-01T00:00:00Z',
      finetuning_type: 'lora' as const,
      base_model: `nvidia/${name.replace('-instruct', '')}`,
      adapters: [
        {
          name: `${name}-adapter-v2`,
          fileset: `nvidia/${name}-v2`,
          finetuning_type: 'lora' as const,
          created_at: '2026-02-15T00:00:00Z',
        },
        {
          name: `${name}-adapter-v1`,
          fileset: `nvidia/${name}-v1`,
          finetuning_type: 'lora' as const,
          created_at: '2026-01-10T00:00:00Z',
        },
      ],
    })),
  },
  {
    workspace: 'meta',
    models: ['llama-3.1-8b', 'llama-3.1-70b', 'llama-3.2-1b', 'llama-3.2-3b', 'llama-3.3-70b'].map(
      (name, i) => ({
        id: `meta-${i}`,
        name,
        workspace: 'meta',
        created_at: '2025-09-01T00:00:00Z',
        updated_at: '2025-09-01T00:00:00Z',
      })
    ),
  },
  {
    workspace: 'mistral',
    models: ['mistral-7b-instruct', 'mistral-8x7b-instruct', 'mistral-large', 'mistral-small'].map(
      (name, i) => ({
        id: `mistral-${i}`,
        name,
        workspace: 'mistral',
        created_at: '2025-07-01T00:00:00Z',
        updated_at: '2025-07-01T00:00:00Z',
        finetuning_type: 'lora' as const,
        base_model: `mistral/${name.replace('-instruct', '')}`,
      })
    ),
  },
  {
    workspace: 'qwen',
    models: [
      'qwen-2.5-7b-instruct',
      'qwen-2.5-14b-instruct',
      'qwen-2.5-32b-instruct',
      'qwen-2.5-72b-instruct',
      'qwen-2.5-coder-7b-instruct',
      'qwen-2.5-coder-32b-instruct',
    ].map((name, i) => ({
      id: `qwen-${i}`,
      name,
      workspace: 'qwen',
      created_at: '2025-06-01T00:00:00Z',
      updated_at: '2025-06-01T00:00:00Z',
      finetuning_type: 'lora' as const,
      base_model: `qwen/${name.replace('-instruct', '')}`,
    })),
  },
  {
    workspace: 'deepseek',
    models: [
      'deepseek-r1-distill-llama-8b',
      'deepseek-r1-distill-llama-70b',
      'deepseek-r1-distill-qwen-7b',
      'deepseek-r1-distill-qwen-32b',
    ].map((name, i) => ({
      id: `ds-${i}`,
      name,
      workspace: 'deepseek',
      created_at: '2025-05-01T00:00:00Z',
      updated_at: '2025-05-01T00:00:00Z',
      finetuning_type: 'lora' as const,
      base_model: `deepseek/${name}`,
    })),
  },
];

const ManyModelsRender = (args: ModelSelectV2Props) => {
  const [value, setValue] = useState<ModelSelection | null>(null);
  return (
    <ModelSelectV2
      {...args}
      value={value}
      onValueChange={setValue}
      groups={manyModelsGroups}
      showModelTypeToggle
    />
  );
};

export const ManyModels: Story = {
  render: ManyModelsRender,
};

const HideAdaptersRender = (args: ModelSelectV2Props) => {
  const [value, setValue] = useState<ModelSelection | null>(null);
  return (
    <ModelSelectV2
      {...args}
      value={value}
      onValueChange={setValue}
      groups={manyModelsGroups}
      hideAdapters
    />
  );
};

export const HideAdapters: Story = {
  render: HideAdaptersRender,
};

const allProgressiveModels = manyModelsGroups.flatMap((group) => group.models);
const PROGRESSIVE_PAGE_SIZE = 6;
const PROGRESSIVE_LATENCY_MS = 600;

/**
 * The paged contract `useModelSearch` implements, stubbed with a local list: the filter box
 * reports its debounced value instead of filtering in place, and pages arrive as the list scrolls.
 */
const ProgressiveRender = (args: ModelSelectV2Props) => {
  const [value, setValue] = useState<ModelSelection | null>(null);
  const [search, setSearch] = useState('');
  const [pageCount, setPageCount] = useState(1);
  const [isLoadingMore, setIsLoadingMore] = useState(false);

  const matches = allProgressiveModels.filter((model) =>
    (model.name ?? '').toLowerCase().includes(search.toLowerCase())
  );
  const loaded = matches.slice(0, pageCount * PROGRESSIVE_PAGE_SIZE);
  const hasMore = loaded.length < matches.length;

  const handleSearchChange = (next: string) => {
    setSearch(next);
    setPageCount(1);
  };

  const handleLoadMore = async () => {
    setIsLoadingMore(true);
    await new Promise((resolve) => setTimeout(resolve, PROGRESSIVE_LATENCY_MS));
    setPageCount((count) => count + 1);
    setIsLoadingMore(false);
  };

  return (
    <ModelSelectV2
      {...args}
      value={value}
      onValueChange={setValue}
      groups={groupByWorkspace(loaded)}
      onSearchChange={handleSearchChange}
      onLoadMore={handleLoadMore}
      hasMore={hasMore}
      isLoadingMore={isLoadingMore}
      doneLoadingMessage="No more models"
      hideAdapters
      fullWidth
    />
  );
};

export const ProgressiveLoading: Story = {
  render: ProgressiveRender,
};
