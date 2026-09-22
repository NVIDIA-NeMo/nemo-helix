// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type Adapter,
  FinetuningType,
  type ModelEntity,
} from '@nemo/sdk/generated/platform/schema';
import { ModelPanel } from '@studio/components/sidePanels/ModelPanels/ModelPanel';
import { mockUseModelChatAvailability } from '@studio/tests/util/mockUseModelChatAvailability';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';

// Capture ModelChat's props so the request identity can be asserted without a chat flow.
const modelChatSpy = vi.fn();
vi.mock('@studio/components/ModelChat', () => ({
  ModelChat: (props: Record<string, unknown>) => {
    modelChatSpy(props);
    return <div data-testid="mock-model-chat" />;
  },
}));

const minimalModelEntity: ModelEntity = {
  id: 'model-test-id',
  name: 'test-model',
  workspace: 'my-workspace',
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-01T00:00:00Z',
  api_endpoint: {
    url: 'http://localhost:8000/v1',
    model_id: 'test-model',
  },
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

describe('ModelPanel', () => {
  it('renders empty state when no model is provided', () => {
    render(
      <TestProviders>
        <ModelPanel open onOpenChange={() => {}} />
      </TestProviders>
    );

    expect(screen.getByText('Awaiting Model Selection')).toBeInTheDocument();
    expect(screen.getByText('Select a base model to get started')).toBeInTheDocument();
  });

  it('renders model name in heading when model is provided', () => {
    render(
      <TestProviders>
        <ModelPanel model={minimalModelEntity} open onOpenChange={() => {}} />
      </TestProviders>
    );

    expect(screen.getByRole('heading', { name: 'test-model' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Model Details' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Chat Playground' })).toBeInTheDocument();
  });

  it('renders Model Details tab content when model is provided', () => {
    render(
      <TestProviders>
        <ModelPanel model={minimalModelEntity} open onOpenChange={() => {}} />
      </TestProviders>
    );

    expect(screen.getByText('Base Model Parameters')).toBeInTheDocument();
    expect(screen.getByText('4096')).toBeInTheDocument();
    expect(screen.getByText('500 million')).toBeInTheDocument();
  });

  describe('chat playground', () => {
    const adapter: Adapter = {
      name: 'pirate-voice',
      workspace: 'other-workspace',
      fileset: 'other-workspace/pirate-voice',
      finetuning_type: FinetuningType.lora,
    };

    beforeEach(() => {
      modelChatSpy.mockClear();
    });

    const renderChat = (props: { adapter?: Adapter } = {}) =>
      render(
        <TestProviders>
          <MemoryRouter>
            <ModelPanel
              model={minimalModelEntity}
              defaultTab="chat-playground"
              open
              onOpenChange={() => {}}
              {...props}
            />
          </MemoryRouter>
        </TestProviders>
      );

    // The adapter used to be reached through the provider proxy, which bypasses every
    // VM-attached middleware (guardrails, Switchyard routing, translate). The gateway
    // now resolves the composite through the base model's VM, so the default base URL
    // is the correct — and only — route.
    it('sends an adapter as a composite name with no base URL override', () => {
      renderChat({ adapter });

      expect(modelChatSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'test-model&adapters/other-workspace/pirate-voice',
          workspace: 'my-workspace',
        })
      );
      expect(modelChatSpy.mock.calls.at(-1)?.[0]).not.toHaveProperty('baseURL');
    });

    it('sends a plain model as its bare name', () => {
      renderChat();

      expect(modelChatSpy).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'test-model', workspace: 'my-workspace' })
      );
    });

    // A READY base deployment that has not loaded the adapter is its own state: the
    // generic "Chat Unavailable" would tell the user to deploy something already up.
    it('keeps the distinct empty state for an adapter no provider serves', () => {
      mockUseModelChatAvailability({ isAdapterUnserved: true, isChatAvailable: false });

      renderChat({ adapter });

      expect(screen.getByText('Adapter is not currently served')).toBeInTheDocument();
      expect(
        screen.getByText(/It becomes available once the deployment serving its base model/)
      ).toBeInTheDocument();
      // The chat surface is replaced outright, not merely disabled.
      expect(modelChatSpy).not.toHaveBeenCalled();
    });
  });
});
