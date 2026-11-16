// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RoutingTestPanel } from '@studio/routes/agents/AgentOptimizationDetailRoute/switchyard/RoutingTestPanel';
import { fireEvent, render, screen } from '@studio/tests/util/render';

const modelChatSpy = vi.fn();

vi.mock('@studio/components/ModelChat', () => ({
  ModelChat: (props: Record<string, unknown>) => {
    modelChatSpy(props);
    return <div data-testid="mock-model-chat" />;
  },
}));

const route = {
  virtualModel: 'support-bot-1a2b3c4d',
  configType: 'random_routing',
  capable: 'default/frontier',
  efficient: 'default/nano',
  judge: null,
  config: { strong_probability: 0.8 },
};

describe('RoutingTestPanel — inference params', () => {
  beforeEach(() => modelChatSpy.mockClear());

  it('starts from the chat defaults with a 1024-token limit', () => {
    render(<RoutingTestPanel workspace="default" route={route} />);

    expect(modelChatSpy).toHaveBeenLastCalledWith(
      expect.objectContaining({
        model: route.virtualModel,
        promptData: { inference_params: { temperature: 0.7, max_tokens: 1024 } },
      })
    );
  });

  it('sends the params the user sets, without offering top p', async () => {
    render(<RoutingTestPanel workspace="default" route={route} />);
    const trigger = screen.getByTestId('params-dropdown-trigger');
    fireEvent.pointerDown(trigger, { button: 0 });
    fireEvent.click(trigger);
    await screen.findByTestId('params-dropdown-content');

    expect(screen.queryByText('Top P')).not.toBeInTheDocument();
    const [temperature] = screen.getAllByRole('spinbutton');
    fireEvent.change(temperature, { target: { value: '1.2' } });

    expect(modelChatSpy).toHaveBeenLastCalledWith(
      expect.objectContaining({
        promptData: { inference_params: { temperature: 1.2, max_tokens: 1024 } },
      })
    );
  });
});
