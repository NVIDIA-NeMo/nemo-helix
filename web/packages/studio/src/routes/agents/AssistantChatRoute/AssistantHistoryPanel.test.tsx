// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AssistantHistoryPanel } from '@studio/routes/agents/AssistantChatRoute/AssistantHistoryPanel';
import { render, screen } from '@studio/tests/util/render';
import { waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const mocks = vi.hoisted(() => ({
  deleteAssistantSessionHistory: vi.fn(),
  listAssistantHistorySessions: vi.fn(),
}));

vi.mock('@studio/routes/agents/AssistantChatRoute/api', () => ({
  deleteAssistantSessionHistory: mocks.deleteAssistantSessionHistory,
  getAssistantHistorySessionsQueryKey: (workspace: string) => [
    'assistant',
    'history',
    'sessions',
    workspace,
  ],
  listAssistantHistorySessions: mocks.listAssistantHistorySessions,
}));

describe('AssistantHistoryPanel', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    mocks.listAssistantHistorySessions.mockResolvedValue([]);
    mocks.deleteAssistantSessionHistory.mockResolvedValue(undefined);
  });

  it('starts history collapsed and expands it', async () => {
    const user = userEvent.setup();
    render(
      <AssistantHistoryPanel
        activeSessionId="session-1"
        onNewChat={vi.fn()}
        onSelectSession={vi.fn()}
      />
    );

    const historyButton = screen.getByRole('button', { name: 'Expand All Chats' });
    expect(historyButton).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('button', { name: 'New chat' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Expand Skills' })).not.toBeInTheDocument();

    await user.click(historyButton);

    expect(screen.getByRole('button', { name: 'Collapse All Chats' })).toHaveAttribute(
      'aria-expanded',
      'true'
    );
    expect(screen.getByRole('button', { name: 'New chat' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'All Chats' })).toHaveClass('min-h-0', 'flex-1');
  });

  it('renders history sessions and keeps selection working', async () => {
    const user = userEvent.setup();
    const onNewChat = vi.fn();
    const onSelectSession = vi.fn();
    mocks.listAssistantHistorySessions.mockResolvedValue([
      {
        session_id: 'session-1',
        mtime: Date.now() / 1000,
        first_prompt: 'Review the latest agent work',
        message_count: 2,
        token_count: 100,
        tool_call_count: 1,
        tool_calls: ['Bash'],
        chat_artifacts: {
          selections: [],
          files: [],
          links: [],
          jobs: [],
          tools: [],
        },
      },
    ]);

    const { unmount } = render(
      <AssistantHistoryPanel
        activeSessionId="session-1"
        onNewChat={onNewChat}
        onSelectSession={onSelectSession}
      />
    );

    await user.click(screen.getByRole('button', { name: 'Expand All Chats' }));

    expect(await screen.findByText('Review the latest agent work')).toBeInTheDocument();
    expect(screen.getByText('Bash')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'New chat' }));
    expect(onNewChat).toHaveBeenCalledTimes(1);

    await user.click(
      screen.getByRole('button', { name: 'Open chat Review the latest agent work' })
    );
    expect(onSelectSession).toHaveBeenCalledWith('session-1');

    unmount();
    render(
      <AssistantHistoryPanel
        activeSessionId="session-1"
        onNewChat={onNewChat}
        onSelectSession={onSelectSession}
      />
    );

    expect(screen.getByRole('button', { name: 'Collapse All Chats' })).toHaveAttribute(
      'aria-expanded',
      'true'
    );
  });

  it('shows the summarized title while preserving the full first prompt in the tooltip', async () => {
    const user = userEvent.setup();
    const firstPrompt = 'I want to create an agent that does spam detection for incoming email.';
    mocks.listAssistantHistorySessions.mockResolvedValue([
      {
        session_id: 'session-1',
        mtime: Date.now() / 1000,
        title: 'Create Spam Detector Agent',
        first_prompt: firstPrompt,
        message_count: 2,
        token_count: 100,
        tool_call_count: 0,
        tool_calls: [],
        chat_artifacts: {
          selections: [],
          files: [],
          links: [],
          jobs: [],
          tools: [],
        },
      },
    ]);

    render(
      <AssistantHistoryPanel
        activeSessionId="session-1"
        onNewChat={vi.fn()}
        onSelectSession={vi.fn()}
      />
    );

    await user.click(screen.getByRole('button', { name: 'Expand All Chats' }));

    const sessionButton = await screen.findByRole('button', {
      name: 'Open chat Create Spam Detector Agent',
    });

    expect(sessionButton).toHaveAttribute('title', expect.stringContaining(firstPrompt));
    expect(screen.queryByText(firstPrompt)).not.toBeInTheDocument();
  });

  it('confirms deletion and starts a new chat when deleting the active session', async () => {
    const user = userEvent.setup();
    const onNewChat = vi.fn();
    mocks.listAssistantHistorySessions.mockResolvedValue([
      {
        session_id: 'session-1',
        mtime: Date.now() / 1000,
        title: 'Private agent work',
        first_prompt: 'Help me with private agent work',
        message_count: 1,
        token_count: 0,
        tool_call_count: 0,
        tool_calls: [],
        chat_artifacts: {
          selections: [],
          files: [],
          links: [],
          jobs: [],
          tools: [],
        },
      },
    ]);

    render(
      <AssistantHistoryPanel
        activeSessionId="session-1"
        workspace="team-a"
        onNewChat={onNewChat}
        onSelectSession={vi.fn()}
      />
    );
    await user.click(screen.getByRole('button', { name: 'Expand All Chats' }));
    await user.click(await screen.findByRole('button', { name: 'Delete chat Private agent work' }));

    expect(
      screen.getByRole('dialog', { name: "Delete chat 'Private agent work'?" })
    ).toBeInTheDocument();
    expect(
      screen.getByText("The chat and its messages are deleted and can't be restored.")
    ).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Delete Chat' }));

    await waitFor(() =>
      expect(mocks.deleteAssistantSessionHistory).toHaveBeenCalledWith('session-1', 'team-a')
    );
    expect(onNewChat).toHaveBeenCalledTimes(1);
  });

  it('renders job artifacts as Studio links', () => {
    render(
      <AssistantHistoryPanel
        activeSessionId="session-1"
        artifacts={{
          workspace: 'default',
          selections: [{ label: 'Environment', value: 'production' }],
          files: [{ action: 'Wrote', path: 'agents/beach-finder.yml' }],
          links: [],
          jobs: [
            {
              name: 'agent-eval-1',
              job_type: 'agent_evaluation',
              source: 'evaluator',
            },
          ],
          tools: ['Bash'],
        }}
        onNewChat={vi.fn()}
        onSelectSession={vi.fn()}
      />
    );

    expect(screen.getByText('Jobs')).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Chat artifacts' })).toHaveClass(
      'overflow-hidden',
      'rounded',
      'border',
      'shrink-0',
      'bg-surface-base',
      'dark:bg-surface-raised'
    );
    expect(screen.queryByText('Workspace')).not.toBeInTheDocument();
    expect(screen.getByText('beach-finder.yml')).toBeInTheDocument();
    expect(screen.getAllByRole('separator')).toHaveLength(3);
    expect(screen.getByRole('link', { name: /agent-eval-1/ })).toHaveAttribute(
      'href',
      '/workspaces/default/agents/evaluations/agent-eval-1'
    );
  });

  it('does not treat workspace metadata as a visible chat artifact', () => {
    render(
      <AssistantHistoryPanel
        activeSessionId="session-1"
        artifacts={{
          workspace: 'default',
          selections: [],
          files: [],
          links: [],
          jobs: [],
          tools: [],
        }}
        onNewChat={vi.fn()}
        onSelectSession={vi.fn()}
      />
    );

    expect(screen.queryByText('Workspace')).not.toBeInTheDocument();
    expect(screen.getByText('No artifacts yet')).toBeInTheDocument();
  });

  it('omits empty artifact sections and their dividers', () => {
    render(
      <AssistantHistoryPanel
        activeSessionId="session-1"
        artifacts={{
          selections: [],
          files: [],
          links: [],
          jobs: [{ name: 'agent-eval-1' }],
          tools: ['Bash'],
        }}
        onNewChat={vi.fn()}
        onSelectSession={vi.fn()}
      />
    );

    expect(screen.queryByText('Selections')).not.toBeInTheDocument();
    expect(screen.getByText('Jobs')).toBeInTheDocument();
    expect(screen.getByText('Tools')).toBeInTheDocument();
    expect(screen.getAllByRole('separator')).toHaveLength(1);
  });

  it('ignores selections with whitespace-only values', () => {
    render(
      <AssistantHistoryPanel
        activeSessionId="session-1"
        artifacts={{
          selections: [{ label: 'Environment', value: ' ' }],
          files: [],
          links: [],
          jobs: [],
          tools: [],
        }}
        onNewChat={vi.fn()}
        onSelectSession={vi.fn()}
      />
    );

    expect(screen.queryByText('Selections')).not.toBeInTheDocument();
    expect(screen.getByText('No artifacts yet')).toBeInTheDocument();
  });
});
