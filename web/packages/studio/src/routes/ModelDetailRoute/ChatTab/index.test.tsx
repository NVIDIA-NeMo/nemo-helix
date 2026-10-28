// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import type { ComparePlaygroundProps } from '@studio/components/ModelCompareChat/ComparePlayground';
import { useModelChatAvailability } from '@studio/hooks/useModelChatAvailability';
import { ChatTab } from '@studio/routes/ModelDetailRoute/ChatTab';
import { render, screen } from '@studio/tests/util/render';

vi.mock('@studio/hooks/useModelChatAvailability');
vi.mock('@studio/components/ModelChatTab', () => ({ ModelChatTab: () => <div>plain chat</div> }));
vi.mock('@studio/components/ModelCompareChat/ComparePlayground', () => ({
  ComparePlayground: ({ models }: ComparePlaygroundProps) => (
    <ul>
      {models.map((m) => (
        <li key={m.id}>
          {m.modelURN} {m.adapter} {m.locked ? 'locked' : ''} {m.promptData?.system_prompt}
        </li>
      ))}
    </ul>
  ),
}));

const model = { name: 'my-model', workspace: 'ws' } as ModelEntity;

const mockAvailability = (availability: Partial<ReturnType<typeof useModelChatAvailability>>) => {
  vi.mocked(useModelChatAvailability).mockReturnValue({
    modelChatStatus: 'enabled',
    isLoading: false,
    isAdapterUnserved: false,
    ...availability,
  } as ReturnType<typeof useModelChatAvailability>);
};

describe('ChatTab', () => {
  it('locks the model in as the first playground panel', () => {
    mockAvailability({});
    render(<ChatTab model={model} adapter={null} />);

    expect(screen.getByRole('listitem')).toHaveTextContent('ws/my-model locked');
  });

  it('hands the saved prompt of the model to the baseline panel', () => {
    mockAvailability({});
    render(
      <ChatTab
        model={{ ...model, prompt: { system_prompt: 'Be terse.' } } as ModelEntity}
        adapter={null}
      />
    );

    expect(screen.getByRole('listitem')).toHaveTextContent('locked Be terse.');
  });

  it('locks in the adapter on an adapter page', () => {
    mockAvailability({});
    render(<ChatTab model={model} adapter={{ name: 'my-adapter' } as Adapter} />);

    expect(screen.getByRole('listitem')).toHaveTextContent('ws/my-model my-adapter locked');
  });

  it('falls back to the plain chat until the model can be chatted with', () => {
    mockAvailability({ modelChatStatus: 'disabled' });
    render(<ChatTab model={model} adapter={null} />);

    expect(screen.getByText('plain chat')).toBeInTheDocument();
    expect(screen.queryByRole('listitem')).not.toBeInTheDocument();
  });

  it('falls back to the plain chat while an adapter is not served', () => {
    mockAvailability({ isAdapterUnserved: true });
    render(<ChatTab model={model} adapter={{ name: 'my-adapter' } as Adapter} />);

    expect(screen.getByText('plain chat')).toBeInTheDocument();
  });
});
