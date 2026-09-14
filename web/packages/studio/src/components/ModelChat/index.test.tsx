// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getEntityReference } from '@nemo/common/src/namedEntity';
import { entityStoreBaseModel1 } from '@studio/mocks/entity-store/models';
import { render, renderRoute, screen } from '@studio/tests/util/render';

import { ModelChat } from '.';

vi.mock('@nemo/common/src/hooks/useChatCompletion', () => ({
  useChatCompletion: () => ({
    mutateAsync: vi.fn(),
  }),
}));

// Deployments are a preview flag, off by default (`previewFlag` defaults to
// false), so the deploy CTA renders nothing unless the flag is on.
vi.mock('@studio/constants/environment', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@studio/constants/environment')>();
  return {
    ...actual,
    DEPLOYMENTS_ENABLED: true,
  };
});

describe('ModelChat', () => {
  const modelName = getEntityReference(entityStoreBaseModel1);

  it('shows the model name in the composer placeholder', async () => {
    render(<ModelChat model={modelName} />);

    expect(await screen.findByPlaceholderText(`Message ${modelName}`)).toBeInTheDocument();
  });

  it('renders provided initialMessages in the thread', async () => {
    render(
      <ModelChat
        model={modelName}
        initialMessages={[
          { role: 'user', content: 'Tell me a story' },
          { role: 'assistant', content: 'Once upon a time...' },
        ]}
      />
    );

    expect(await screen.findByText('Tell me a story')).toBeInTheDocument();
    expect(await screen.findByText('Once upon a time...')).toBeInTheDocument();
  });

  it('disables the composer when modelChatStatus is not enabled', async () => {
    render(<ModelChat model={modelName} modelChatStatus="pending" />);

    expect(await screen.findByRole('textbox', { name: /Task prompt/i })).toBeDisabled();
    expect(await screen.findByText('Model Deployment in Progress')).toBeInTheDocument();
  });

  it('offers a deploy CTA when the chat is disabled and a model ref is given', async () => {
    renderRoute(
      <ModelChat model={modelName} modelChatStatus="disabled" deployModelRef="default/my-model" />
    );

    expect(await screen.findByText('Chat Unavailable')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Deploy this model' })).toBeInTheDocument();
  });

  it('omits the deploy CTA when no model ref is given', async () => {
    renderRoute(<ModelChat model={modelName} modelChatStatus="disabled" />);

    expect(await screen.findByText('Chat Unavailable')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^Deploy/ })).not.toBeInTheDocument();
  });

  it('uses the caller-supplied label (adapter chats deploy the base model)', async () => {
    renderRoute(
      <ModelChat
        model={modelName}
        modelChatStatus="disabled"
        deployModelRef="default/base-model"
        deployModelLabel="Deploy base model"
      />
    );

    expect(await screen.findByRole('button', { name: 'Deploy base model' })).toBeInTheDocument();
  });

  it('does not offer the CTA while a deployment is pending', async () => {
    renderRoute(
      <ModelChat model={modelName} modelChatStatus="pending" deployModelRef="default/my-model" />
    );

    expect(await screen.findByText('Model Deployment in Progress')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^Deploy/ })).not.toBeInTheDocument();
  });
});
