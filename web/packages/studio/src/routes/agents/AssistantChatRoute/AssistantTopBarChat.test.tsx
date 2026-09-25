// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AssistantTopBarChat } from '@studio/routes/agents/AssistantChatRoute/AssistantTopBarChat';
import { mockUseParams } from '@studio/tests/util/mockUseParams';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router';

const mocks = vi.hoisted(() => ({
  chat: {
    decisionRequest: null,
    inputRequest: null,
    isRunning: false,
  } as { decisionRequest: unknown; inputRequest: unknown; isRunning: boolean },
  startNewChat: vi.fn(),
}));

vi.mock('@studio/constants/environment', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@studio/constants/environment')>();
  return {
    ...actual,
    ASSISTANT_STUDIO_ENABLED: true,
  };
});

vi.mock('@studio/routes/agents/AssistantChatRoute/context/useAssistantChatContext', () => ({
  useAssistantChatContext: () => ({
    chat: mocks.chat,
    loadStatus: 'idle',
    loadSession: vi.fn(),
    startNewChat: mocks.startNewChat,
  }),
}));

vi.mock('@studio/routes/agents/AssistantChatRoute/AssistantChatThread', () => ({
  AssistantChatThread: () => (
    <div data-testid="compact-chat-thread">
      <a href="/workspaces/default/jobs/job-1" onClick={(event) => event.preventDefault()}>
        Job details
      </a>
    </div>
  ),
}));

const WORKSPACE = 'default';

const LocationProbe = () => {
  const location = useLocation();
  return <div data-testid="pathname">{location.pathname}</div>;
};

const getTopBarChatElement = (initialPath = `/workspaces/${WORKSPACE}/jobs`) => (
  <TestProviders>
    <MemoryRouter initialEntries={[initialPath]}>
      <LocationProbe />
      <Routes>
        <Route path="/workspaces/:workspace/*" element={<AssistantTopBarChat />} />
      </Routes>
    </MemoryRouter>
  </TestProviders>
);

const renderTopBarChat = (initialPath = `/workspaces/${WORKSPACE}/jobs`) =>
  render(getTopBarChatElement(initialPath));

describe('AssistantTopBarChat', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseParams({ workspace: WORKSPACE });
    mocks.chat.isRunning = false;
    mocks.chat.decisionRequest = null;
    mocks.chat.inputRequest = null;
  });

  it('opens and closes the compact chat from the top bar icon', async () => {
    renderTopBarChat();
    const user = userEvent.setup();
    const trigger = screen.getByRole('button', { name: 'Open NeMo Assistant chat' });

    await user.click(trigger);
    expect(await screen.findByTestId('compact-chat-thread')).toBeVisible();

    await user.click(screen.getByRole('button', { name: 'Close NeMo Assistant chat' }));
    await waitFor(() =>
      expect(screen.queryByTestId('compact-chat-thread')).not.toBeInTheDocument()
    );
  });

  it('darkens the screen with a backdrop and closes the chat when it is clicked', async () => {
    renderTopBarChat();
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Open NeMo Assistant chat' }));
    const backdrop = await screen.findByTestId('assistant-chat-backdrop');
    expect(backdrop).toBeVisible();

    await user.click(backdrop);

    await waitFor(() =>
      expect(screen.queryByTestId('assistant-chat-backdrop')).not.toBeInTheDocument()
    );
  });

  it('navigates to the main chat from the popout header', async () => {
    renderTopBarChat();
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Open NeMo Assistant chat' }));
    await user.click(await screen.findByRole('button', { name: 'Open in main chat' }));

    await waitFor(() => expect(screen.getByTestId('pathname').textContent).toContain('assistant'));
  });

  it('starts a new compact chat from the popout header', async () => {
    renderTopBarChat();
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Open NeMo Assistant chat' }));
    await user.click(await screen.findByRole('button', { name: /New/i }));

    expect(mocks.startNewChat).toHaveBeenCalledOnce();
  });

  it('closes the compact chat when a chat link is clicked', async () => {
    renderTopBarChat();
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Open NeMo Assistant chat' }));
    expect(await screen.findByTestId('compact-chat-thread')).toBeVisible();

    await user.click(screen.getByRole('link', { name: 'Job details' }));
    await waitFor(() =>
      expect(screen.queryByTestId('compact-chat-thread')).not.toBeInTheDocument()
    );
  });

  it('shows a thinking indicator while the agent is running', () => {
    mocks.chat.isRunning = true;

    renderTopBarChat();

    expect(screen.getAllByTestId('assistant-thinking-dot')).toHaveLength(3);
    expect(screen.queryByTestId('assistant-unread-indicator')).not.toBeInTheDocument();
  });

  it('surfaces an unread badge after a response finishes while closed', () => {
    const view = renderTopBarChat();

    expect(screen.queryByTestId('assistant-unread-indicator')).not.toBeInTheDocument();

    mocks.chat.isRunning = true;
    view.rerender(getTopBarChatElement());
    mocks.chat.isRunning = false;
    view.rerender(getTopBarChatElement());

    expect(screen.getByTestId('assistant-unread-indicator')).toBeInTheDocument();
  });

  it('shows the attention badge instead of the thinking dots while awaiting user input', () => {
    mocks.chat.isRunning = true;
    mocks.chat.inputRequest = { requestId: 'request-1', kind: 'dataset_file', input: {} };

    renderTopBarChat();

    expect(screen.getByTestId('assistant-unread-indicator')).toBeInTheDocument();
    expect(screen.queryByTestId('assistant-thinking-indicator')).not.toBeInTheDocument();
  });
});
