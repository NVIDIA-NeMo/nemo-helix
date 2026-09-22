// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ModelChatPanel } from '@studio/components/ModelChatPanel';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render } from '@testing-library/react';
import { MemoryRouter } from 'react-router';

// Capture the props that ModelChat receives so we can assert on workspace/model
// routing without spinning up a real chat flow.
const modelChatSpy = vi.fn();

vi.mock('@studio/components/ModelChat', () => ({
  ModelChat: (props: Record<string, unknown>) => {
    modelChatSpy(props);
    return <div data-testid="mock-model-chat" />;
  },
}));

// The model select fetches its own options; its internals are not what we're testing here.
vi.mock('@nemo/common/src/components/ModelSelectV2', () => ({
  WorkspaceModelSelect: () => <div data-testid="mock-model-select" />,
}));

const renderPanel = (modelURN: string | null, adapter: string | null = null) => {
  return render(
    <TestProviders>
      <MemoryRouter>
        <ModelChatPanel
          panel={{
            id: 0,
            collapsed: false,
            modelURN,
            adapter,
            roleColor: 'baseline',
            roleLabel: 'Baseline',
            isSinglePanel: true,
            locked: false,
          }}
          fallbackWorkspace="route-workspace"
          onToggle={vi.fn()}
          onRemove={vi.fn()}
          onModelChange={vi.fn()}
        />
      </MemoryRouter>
    </TestProviders>
  );
};

describe('ModelChatPanel — URN routing', () => {
  beforeEach(() => {
    modelChatSpy.mockClear();
  });

  it("routes inference to the model's own workspace (not the route workspace)", () => {
    renderPanel('nvidia/llama-70b');

    expect(modelChatSpy).toHaveBeenCalledWith(
      expect.objectContaining({ workspace: 'nvidia', model: 'llama-70b' })
    );
  });

  it('picks the correct workspace even when two models share the same name', () => {
    // A name-based lookup would have silently bound this panel to whichever workspace's model
    // came first in the list. With URNs end-to-end, the workspace in the URN is used.
    renderPanel('abacusai/llama-70b');

    expect(modelChatSpy).toHaveBeenCalledWith(
      expect.objectContaining({ workspace: 'abacusai', model: 'llama-70b' })
    );
  });

  // An unresolved adapter used to leave the panel enabled and pointed at the base model, so the
  // trigger read "llama-70b / support-v1" while the request went to the base weights.
  it('never sends to the base model while a selected adapter is unresolved', () => {
    renderPanel('nvidia/llama-70b', 'support-v1');

    expect(modelChatSpy).not.toHaveBeenCalledWith(expect.objectContaining({ model: 'llama-70b' }));
    expect(modelChatSpy).toHaveBeenLastCalledWith(
      expect.objectContaining({ model: '', disabled: true })
    );
  });

  it('falls back to the route workspace and disables chat when no model is assigned', () => {
    renderPanel(null);
    // ModelChat still renders, but disabled and showing an empty state; with no
    // model URN it uses the route fallback workspace and an empty model id.
    expect(modelChatSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        workspace: 'route-workspace',
        model: '',
        disabled: true,
        emptyState: expect.objectContaining({ slotHeading: expect.any(String) }),
      })
    );
  });
});
