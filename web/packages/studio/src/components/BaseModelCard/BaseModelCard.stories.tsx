/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import type { ModelEntity, ModelSpec } from '@nemo/sdk/generated/platform/schema';
import type { Meta, StoryObj } from '@storybook/react';
import { BaseModelCard } from '@studio/components/BaseModelCard';

const meta = {
  component: BaseModelCard,
  title: 'Components/BaseModelCard',
  args: {
    isChatAvailable: true,
  },
  decorators: [
    (Story, context) => (
      // eslint-disable-next-line no-restricted-syntax
      <div style={{ width: context.parameters.containerWidth ?? '320px' }}>
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof BaseModelCard>;

export default meta;
type Story = StoryObj<typeof meta>;

/** Build a minimal ModelEntity — only the fields the card actually reads. */
const makeModel = (
  fields: Pick<ModelEntity, 'name' | 'workspace'> & {
    description?: string;
    spec?: Partial<ModelSpec>;
    model_providers?: string[];
    fileset?: string;
  }
): ModelEntity => ({
  id: Math.random().toString(36).substring(2, 15),
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-01T00:00:00Z',
  name: fields.name,
  workspace: fields.workspace,
  description: fields.description,
  spec: fields.spec as ModelSpec | undefined,
  model_providers: fields.model_providers,
  fileset: fields.fileset,
});

/**
 * A well-known Meta model — shows creator icon, description, customizable badge,
 * and all spec fields.
 */
export const Default: Story = {
  args: {
    model: makeModel({
      name: 'llama-3.1-8b-instruct',
      workspace: 'meta',
      description:
        'Llama-3.1-8b is a large language AI model optimized for multilingual dialogue uses.',
      spec: {
        base_num_parameters: 8_000_000_000,
        context_size: 8192,
        is_chat: true,
      },
      model_providers: ['default/nvidia-build'],
    }),
  },
};

/** An NVIDIA Nemotron model with multiple fine-tune options. */
export const NvidiaModel: Story = {
  args: {
    model: makeModel({
      name: 'nemotron-nano-llama-3.1-8b',
      workspace: 'nvidia',
      description:
        'Llama-3.1 Nemotron Nano 8B v1 is a compact, instruction-tuned model for efficient customization and deployment.',
      spec: {
        base_num_parameters: 8_000_000_000,
        context_size: 4096,
        is_chat: true,
      },
      model_providers: ['default/nvidia-build', 'default/build'],
    }),
  },
};

/** No spec data at all — the footer specs section is empty. */
export const NoSpecs: Story = {
  args: {
    model: makeModel({
      name: 'custom-model-v1',
      workspace: 'default',
      description: 'A custom model with no specification data available.',
    }),
  },
};

/** No description — only name, creator, and specs are visible. */
export const NoDescription: Story = {
  args: {
    model: makeModel({
      name: 'unlisted-model-70b',
      workspace: 'acme',
      spec: {
        base_num_parameters: 70_000_000_000,
        context_size: 8192,
        is_chat: true,
      },
    }),
  },
};

/** A non-chat model — the chat indicator is hidden. */
export const NonChat: Story = {
  args: {
    isChatAvailable: false,
    model: makeModel({
      name: 'phi-4',
      workspace: 'microsoft',
      description:
        "Phi-4 is Microsoft's most advanced small language model, designed to deliver strong reasoning capabilities.",
      spec: {
        base_num_parameters: 14_000_000_000,
        context_size: 16384,
        is_chat: false,
      },
    }),
  },
};

/** Very long name and description to test text truncation behavior. */
export const LongContent: Story = {
  args: {
    model: makeModel({
      name: 'extremely-long-model-name-that-should-be-truncated-in-the-card-header',
      workspace: 'nvidia',
      description:
        'This is an extremely long description that is intended to test how the card handles overflow. It should be clamped to two lines with an ellipsis, keeping the layout consistent with other cards in the grid.',
      spec: {
        base_num_parameters: 405_000_000_000,
        context_size: 131072,
        is_chat: true,
      },
    }),
  },
};

/** A small model with parameters in the millions range. */
export const SmallModel: Story = {
  args: {
    model: makeModel({
      name: 'tiny-llm-v2',
      workspace: 'default',
      description: 'A lightweight model with parameters in the millions range.',
      spec: {
        base_num_parameters: 125_000_000,
        context_size: 512,
        is_chat: false,
      },
    }),
  },
};

const manyProvidersModel = makeModel({
  name: 'multi-provider-model',
  workspace: 'nvidia',
  description: 'A model served by many providers to test tag overflow.',
  spec: {
    base_num_parameters: 8_000_000_000,
    context_size: 4096,
    is_chat: true,
  },
  model_providers: [
    'default/nvidia-build',
    'default/build',
    'default/classify-llm-tutorial-1771886621',
    'default/classify-llm-tutorial-1771886858',
  ],
});

/** Multiple providers. */
export const ManyProviders: Story = {
  args: { model: manyProvidersModel },
};

const manyProvidersProviders = [
  'default/nvidia-build',
  'default/build',
  'default/classify-llm-tutorial-1771886621',
  'default/classify-llm-tutorial-1771886858',
];

const manyProvidersNoIcons = makeModel({
  name: 'multi-provider-model',
  workspace: 'nvidia',
  description: 'A model served by many providers to test tag overflow.',
  model_providers: manyProvidersProviders,
});

const manyProviders1Icon = makeModel({
  name: 'multi-provider-model',
  workspace: 'nvidia',
  description: 'A model served by many providers to test tag overflow.',
  spec: { base_num_parameters: 8_000_000_000 } as ModelSpec,
  model_providers: manyProvidersProviders,
});

const manyProviders2Icons = makeModel({
  name: 'multi-provider-model',
  workspace: 'nvidia',
  description: 'A model served by many providers to test tag overflow.',
  spec: { base_num_parameters: 8_000_000_000, context_size: 4096 } as ModelSpec,
  model_providers: manyProvidersProviders,
});

/** Many providers at varying card widths to visualize overflow behavior. */
export const ManyProvidersWidths: Story = {
  args: { model: manyProvidersModel },
  parameters: { containerWidth: 'auto' },
  render: () => (
    <div className="flex flex-col gap-6">
      <div>
        <p className="text-sm text-secondary mb-2">320px — 0 icons</p>
        {/* eslint-disable-next-line no-restricted-syntax */}
        <div style={{ width: '320px' }}>
          <BaseModelCard model={manyProvidersNoIcons} isChatAvailable />
        </div>
      </div>
      <div>
        <p className="text-sm text-secondary mb-2">320px — 1 icon</p>
        {/* eslint-disable-next-line no-restricted-syntax */}
        <div style={{ width: '320px' }}>
          <BaseModelCard model={manyProviders1Icon} isChatAvailable />
        </div>
      </div>
      <div>
        <p className="text-sm text-secondary mb-2">320px — 2 icons</p>
        {/* eslint-disable-next-line no-restricted-syntax */}
        <div style={{ width: '320px' }}>
          <BaseModelCard model={manyProviders2Icons} isChatAvailable />
        </div>
      </div>
      {[320, 380, 440, 500].map((width) => (
        <div key={width}>
          <p className="text-sm text-secondary mb-2">{width}px — 3 icons</p>
          {/* eslint-disable-next-line no-restricted-syntax */}
          <div style={{ width: `${width}px` }}>
            <BaseModelCard model={manyProvidersModel} isChatAvailable />
          </div>
        </div>
      ))}
    </div>
  ),
};

/** No providers — no tags shown in footer. */
export const NoProviders: Story = {
  args: {
    model: makeModel({
      name: 'undeployed-model',
      workspace: 'default',
      description: 'A model with no providers configured.',
      spec: {
        base_num_parameters: 8_000_000_000,
        context_size: 4096,
        is_chat: true,
      },
      model_providers: [],
    }),
  },
};

const customizationSpec = {
  base_num_parameters: 8_000_000_000,
  context_size: 4096,
  is_chat: true,
};

const fineTuneableOnlyModel = makeModel({
  name: 'fine-tuneable-only',
  workspace: 'meta',
  description: 'Has a fileset, so the Fine-Tuneable badge renders.',
  spec: customizationSpec,
  model_providers: ['default/nvidia-build'],
  fileset: 'meta/llama-checkpoint',
});

/** The fine-tunable customization badge, which renders when the model has a fileset. */
export const CustomizationBadges: Story = {
  args: { model: fineTuneableOnlyModel },
  render: () => (
    <div className="flex flex-col gap-6">
      <div>
        <p className="text-sm text-secondary mb-2">Fine-Tuneable only</p>
        <BaseModelCard model={fineTuneableOnlyModel} isChatAvailable />
      </div>
    </div>
  ),
};
