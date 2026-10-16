// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import {
  ModelDetailOverview,
  ModelParametersAccordion,
} from '@studio/components/sidePanels/ModelPanels/ModelPanel/components';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';

const minimalModelEntity: ModelEntity = {
  id: 'model-test-id',
  name: 'test-model',
  workspace: 'my-workspace',
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-01T00:00:00Z',
  spec: {
    checkpoint_model_name: 'nvidia/test:1.0',
    family: 'llama',
    context_size: 4096,
    base_num_parameters: 500_000_000,
    num_layers: 12,
    hidden_size: 1024,
    num_attention_heads: 16,
    num_kv_heads: 16,
    ffn_hidden_size: 4096,
    vocab_size: 32000,
    tied_embeddings: false,
    gated_mlp: true,
    precision: 'bfloat16',
  },
};

describe('ModelDetailOverview', () => {
  it('renders model name and version badge', () => {
    render(
      <TestProviders>
        <ModelDetailOverview model={minimalModelEntity} />
      </TestProviders>
    );
    expect(screen.getByText('test-model')).toBeInTheDocument();
    expect(screen.getByText('1.0')).toBeInTheDocument();
  });

  it('renders description when provided', () => {
    const modelWithDescription: ModelEntity = {
      ...minimalModelEntity,
      description: 'A test model for unit tests.',
    };
    render(
      <TestProviders>
        <ModelDetailOverview model={modelWithDescription} />
      </TestProviders>
    );
    expect(screen.getByText('A test model for unit tests.')).toBeInTheDocument();
  });

  it('renders description override when overviewProps.description is passed', () => {
    render(
      <TestProviders>
        <ModelDetailOverview
          model={{ ...minimalModelEntity, description: 'Entity description' }}
          description="Override description"
        />
      </TestProviders>
    );
    expect(screen.getByText('Override description')).toBeInTheDocument();
    expect(screen.queryByText('Entity description')).not.toBeInTheDocument();
  });

  it('renders status when overviewProps.status is passed', () => {
    render(
      <TestProviders>
        <ModelDetailOverview model={minimalModelEntity} status="READY" />
      </TestProviders>
    );
    // The status text is rendered twice (visible label + tooltip content);
    // assert at least one is present.
    expect(screen.getAllByText('READY').length).toBeGreaterThan(0);
  });

  it('renders badges when provided', () => {
    render(
      <TestProviders>
        <ModelDetailOverview model={minimalModelEntity} badges={['tool-calling', 'reasoning']} />
      </TestProviders>
    );
    expect(screen.getByText('Tool Calling')).toBeInTheDocument();
    expect(screen.getByText('Reasoning')).toBeInTheDocument();
  });

  it('renders slotActions when provided', () => {
    render(
      <TestProviders>
        <ModelDetailOverview
          model={minimalModelEntity}
          slotActions={<button type="button">Customize</button>}
        />
      </TestProviders>
    );
    expect(screen.getByRole('button', { name: 'Customize' })).toBeInTheDocument();
  });
});

describe('ModelParametersAccordion', () => {
  it('renders Base Model Parameters section with spec data', () => {
    render(
      <TestProviders>
        <ModelParametersAccordion model={minimalModelEntity} />
      </TestProviders>
    );
    expect(screen.getByText('Base Model Parameters')).toBeInTheDocument();
    expect(screen.getByText('4096')).toBeInTheDocument();
    expect(screen.getByText('500 million')).toBeInTheDocument();
    expect(screen.getByText('test-model')).toBeInTheDocument();
    expect(screen.getByText('nvidia/test:1.0')).toBeInTheDocument();
  });

  it('renders Inference section when model has api_endpoint', () => {
    const modelWithEndpoint: ModelEntity = {
      ...minimalModelEntity,
      api_endpoint: {
        url: 'https://api.example.com/v1',
        model_id: 'test-model-id',
        api_key: 'sk-secret',
        format: 'openai',
      },
    };
    render(
      <TestProviders>
        <ModelParametersAccordion model={modelWithEndpoint} />
      </TestProviders>
    );
    expect(screen.getByText('Inference')).toBeInTheDocument();
    expect(screen.getByText('https://api.example.com/v1')).toBeInTheDocument();
    expect(screen.getByText('test-model-id')).toBeInTheDocument();
  });

  it('renders Customization Parameters only for customized models', () => {
    render(
      <TestProviders>
        <ModelParametersAccordion model={minimalModelEntity} />
      </TestProviders>
    );
    expect(screen.queryByText('Customization Parameters')).not.toBeInTheDocument();

    const customizedModel: ModelEntity = {
      ...minimalModelEntity,
      base_model: 'meta/llama-3.2-1b',
      finetuning_type: 'lora_merged',
      created_at: '2025-01-01T00:00:00Z',
    };
    render(
      <TestProviders>
        <ModelParametersAccordion model={customizedModel} />
      </TestProviders>
    );
    expect(screen.getByText('Customization Parameters')).toBeInTheDocument();
    expect(screen.getByText('meta/llama-3.2-1b')).toBeInTheDocument();
    expect(screen.getAllByText('LoRA Merged').length).toBeGreaterThanOrEqual(1);
  });

  it('hides customization details when customization details are disabled', () => {
    const customizedModel: ModelEntity = {
      ...minimalModelEntity,
      base_model: 'meta/llama-3.2-1b',
      finetuning_type: 'lora_merged',
    };

    render(
      <TestProviders>
        <ModelParametersAccordion model={customizedModel} showCustomizationDetails={false} />
      </TestProviders>
    );

    expect(screen.queryByText('Customization Parameters')).not.toBeInTheDocument();
    expect(screen.queryByText('Fine-tune Options')).not.toBeInTheDocument();
    expect(screen.queryByText('Recommended GPUs for Customization')).not.toBeInTheDocument();
  });

  it('renders View Job Details link when customizationJobId is provided', () => {
    const customizedModel: ModelEntity = {
      ...minimalModelEntity,
      base_model: 'meta/llama',
      workspace: 'ws1',
    };
    render(
      <TestProviders>
        <ModelParametersAccordion model={customizedModel} customizationJobId="job-123" />
      </TestProviders>
    );
    const link = screen.getByRole('link', { name: 'View Job Details' });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute('href', expect.stringContaining('job-123'));
  });

  describe('on an adapter row', () => {
    const adapter = {
      name: 'grumpy-purple-gorilla',
      workspace: 'adapter-ws',
      fileset: 'adapter-ws/grumpy-purple-gorilla',
      finetuning_type: 'lora',
      created_at: '2025-01-02T00:00:00Z',
    } as Adapter;
    // A *fine-tuned* base carrying the adapter, so `isCustomizedModel` holds and the
    // Customization section actually renders. With a plain base model it does not, and a
    // link in the wrong section would go unnoticed.
    const fineTunedBase: ModelEntity = {
      ...minimalModelEntity,
      name: 'qwen3-0.6b-sft',
      workspace: 'base-ws',
      base_model: 'qwen/qwen3-0.6b',
      finetuning_type: 'all_weights',
      adapters: [adapter],
    };

    it("links the adapter's run in the adapter's own workspace", () => {
      render(
        <TestProviders>
          <ModelParametersAccordion
            model={fineTunedBase}
            adapter={adapter}
            adapterJobId="lora-job-456"
          />
        </TestProviders>
      );

      // Both sections render, so a misplaced link would be visible.
      expect(screen.getByText('Customization Parameters')).toBeInTheDocument();
      expect(screen.getByText('Adapter Details')).toBeInTheDocument();

      const links = screen.getAllByRole('link', { name: 'View Job Details' });
      expect(links).toHaveLength(1);
      // The adapter's workspace, not the base's: the job lives where its output was
      // registered, and the two can differ.
      expect(links[0]).toHaveAttribute('href', expect.stringMatching(/adapter-ws\/.*lora-job-456/));
    });

    it("renders the model's own run only when the caller supplies it", () => {
      render(
        <TestProviders>
          <ModelParametersAccordion
            model={fineTunedBase}
            adapter={adapter}
            customizationJobId="sft-job-123"
            adapterJobId="lora-job-456"
          />
        </TestProviders>
      );

      // The component is agnostic: given both, it shows both, each in its own section and
      // its own workspace. Suppressing the base's run on an adapter row is the caller's
      // decision -- see CustomizationJobListRoute.
      const hrefs = screen
        .getAllByRole('link', { name: 'View Job Details' })
        .map((link) => link.getAttribute('href'));
      expect(hrefs).toHaveLength(2);
      expect(hrefs).toEqual(
        expect.arrayContaining([
          expect.stringMatching(/base-ws\/.*sft-job-123/),
          expect.stringMatching(/adapter-ws\/.*lora-job-456/),
        ])
      );
    });
  });

  it('renders Prompt section when model has prompt', () => {
    const modelWithPrompt: ModelEntity = {
      ...minimalModelEntity,
      prompt: {
        system_prompt: 'You are a helpful assistant.',
        icl_few_shot_examples: 'Example 1',
      },
    };
    render(
      <TestProviders>
        <ModelParametersAccordion model={modelWithPrompt} />
      </TestProviders>
    );
    expect(screen.getAllByText('Prompt').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('You are a helpful assistant.')).toBeInTheDocument();
  });

  it('renders Artifact Data with deployment and artifactData', () => {
    render(
      <TestProviders>
        <ModelParametersAccordion
          model={{
            ...minimalModelEntity,
            fileset: 'ws/fileset-1',
          }}
          deployment={{
            id: '123',
            created_at: '2025-01-01T00:00:00Z',
            updated_at: '2025-01-01T00:00:00Z',
            name: 'test-model',
            workspace: 'my-workspace',
            status: 'READY',
            entity_version: 1,
            config: 'default',
            config_version: 1,
          }}
          artifactData={{
            backend_engine: 'nemo',
            gpu_architecture: 'Ampere',
            tensor_parallelism: 2,
          }}
        />
      </TestProviders>
    );
    expect(screen.getByText('Artifact Data')).toBeInTheDocument();
    expect(screen.getByText('ws/fileset-1')).toBeInTheDocument();
    expect(screen.getByText('READY')).toBeInTheDocument();
    expect(screen.getByText('nemo')).toBeInTheDocument();
    expect(screen.getByText('Ampere')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument();
    expect(screen.getByText('bfloat16')).toBeInTheDocument();
  });

  it('renders Fine-tune Options from model.finetuning_type', () => {
    const modelWithFinetuning: ModelEntity = {
      ...minimalModelEntity,
      finetuning_type: 'lora',
    };
    render(
      <TestProviders>
        <ModelParametersAccordion model={modelWithFinetuning} />
      </TestProviders>
    );
    expect(screen.getByText('Base Model Parameters')).toBeInTheDocument();
    expect(screen.getAllByText('LoRA').length).toBeGreaterThanOrEqual(1);
  });
});
