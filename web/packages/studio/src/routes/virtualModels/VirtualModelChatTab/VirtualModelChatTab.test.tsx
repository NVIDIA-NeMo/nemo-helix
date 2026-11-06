// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { VirtualModelChatTab } from '@studio/routes/virtualModels/VirtualModelChatTab';
import { fireEvent, render, screen } from '@studio/tests/util/render';

const modelChatSpy = vi.fn();

vi.mock('@studio/components/ModelChat', () => ({
  ModelChat: (props: Record<string, unknown>) => {
    modelChatSpy(props);
    return <div data-testid="mock-model-chat" />;
  },
}));

vi.mock('@studio/hooks/useWorkspaceFromPath', () => ({
  useWorkspaceFromPath: () => 'default',
}));

vi.mock('@studio/util/hooks/useRequiredPathParams', () => ({
  useRequiredPathParams: () => ({ virtualModelName: 'support-router' }),
}));

const openParams = async () => {
  const trigger = screen.getByTestId('params-dropdown-trigger');
  fireEvent.pointerDown(trigger, { button: 0 });
  fireEvent.click(trigger);
  await screen.findByTestId('params-dropdown-content');
};

describe('VirtualModelChatTab', () => {
  beforeEach(() => modelChatSpy.mockClear());

  it('chats with the VirtualModel from the URL, without a model picker', () => {
    render(<VirtualModelChatTab />);

    expect(screen.queryByTestId('model-select-v2-trigger')).not.toBeInTheDocument();
    expect(modelChatSpy).toHaveBeenLastCalledWith(
      expect.objectContaining({
        model: 'support-router',
        workspace: 'default',
        promptData: { inference_params: { temperature: 0.7, max_tokens: 4096 } },
      })
    );
  });

  it('offers only the params chat sends, and sends what the user sets', async () => {
    render(<VirtualModelChatTab />);
    await openParams();

    expect(screen.getByText('Temperature')).toBeInTheDocument();
    expect(screen.getByText('Max Tokens')).toBeInTheDocument();
    expect(screen.queryByText('Top P')).not.toBeInTheDocument();

    const [temperature, maxTokens] = screen.getAllByRole('spinbutton');
    expect(maxTokens).toHaveValue(4096);
    fireEvent.change(temperature, { target: { value: '0' } });

    expect(modelChatSpy).toHaveBeenLastCalledWith(
      expect.objectContaining({
        promptData: { inference_params: { temperature: 0, max_tokens: 4096 } },
      })
    );
  });

  it('resets max tokens to the limit this tab starts with', async () => {
    render(<VirtualModelChatTab />);
    await openParams();

    const maxTokens = screen.getByRole('spinbutton', { name: 'Max Tokens value' });
    fireEvent.change(maxTokens, { target: { value: '256' } });
    fireEvent.click(screen.getByRole('button', { name: 'Reset Max Tokens to default value' }));

    expect(modelChatSpy).toHaveBeenLastCalledWith(
      expect.objectContaining({
        promptData: { inference_params: { temperature: 0.7, max_tokens: 4096 } },
      })
    );
  });

  it('keeps sending the last temperature when its box is cleared', async () => {
    render(<VirtualModelChatTab />);
    await openParams();

    const temperature = screen.getByRole('spinbutton', { name: 'Temperature value' });
    fireEvent.change(temperature, { target: { value: '0.4' } });
    fireEvent.change(temperature, { target: { value: '' } });
    fireEvent.blur(temperature);

    expect(temperature).toHaveValue(0.4);
    expect(modelChatSpy).toHaveBeenLastCalledWith(
      expect.objectContaining({
        promptData: { inference_params: { temperature: 0.4, max_tokens: 4096 } },
      })
    );
  });
});
