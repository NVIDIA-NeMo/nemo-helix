// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { AssistantChatThread } from '@studio/routes/agents/AssistantChatRoute/AssistantChatThread';
import type {
  AssistantChatRuntime,
  StudioNavigationRequest,
} from '@studio/routes/agents/AssistantChatRoute/useAssistantChatRuntime';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { createMemoryRouter, generatePath, RouterProvider } from 'react-router';

const mocks = vi.hoisted(() => ({
  resolveStudioNavigationRequest: vi.fn(),
}));

vi.mock('@assistant-ui/react', () => ({
  AssistantRuntimeProvider: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock('@nemo/common/src/components/AssistantChat/AssistantChatThread', () => ({
  AssistantChatThread: ({
    animateAssistantMessageCompletion,
    composerOverride,
    emptyState,
    hideResetButton,
    minInputRows,
    placeholder,
    showRunningIndicator,
  }: {
    animateAssistantMessageCompletion?: boolean;
    composerOverride?: ReactNode;
    emptyState?: { slotHeading?: string; variant?: string };
    hideResetButton?: boolean;
    minInputRows?: number;
    placeholder?: string;
    showRunningIndicator?: boolean;
  }) => (
    <div
      data-testid="assistant-chat-thread"
      data-animate-message-completion={String(animateAssistantMessageCompletion)}
      data-show-running-indicator={String(showRunningIndicator)}
      data-hide-reset-button={String(hideResetButton)}
      data-min-input-rows={String(minInputRows)}
      data-placeholder={placeholder}
      data-empty-heading={emptyState?.slotHeading}
      data-empty-variant={String(emptyState?.variant)}
    >
      {composerOverride}
    </div>
  ),
}));

const WORKSPACE = 'default';
const CHAT_PATH = generatePath(ROUTES.workspace.assistantChat, { workspace: WORKSPACE });

const makeStudioNavigationRequest = (
  overrides?: Partial<StudioNavigationRequest>
): StudioNavigationRequest => ({
  id: 'guardrails:1',
  prompt: 'Add guardrails to an agent',
  suggestion: {
    id: 'guardrails',
    title: 'Open Guardrails',
    description: 'Studio has a UI for managing NeMo Guardrails configurations.',
    href: '/workspaces/default/guardrails',
  },
  ...overrides,
});

const makeChat = (studioNavigationRequest: StudioNavigationRequest | null) =>
  ({
    artifacts: { selections: [], files: [], links: [], tools: [] },
    decisionChoices: [],
    decisionRequest: null,
    decisionStatus: 'pending',
    handleReset: vi.fn(),
    inputRequest: null,
    inputStatus: 'pending',
    isRunning: false,
    loadSession: vi.fn(),
    resolveDecisionRequest: vi.fn(),
    resolveInputRequest: vi.fn(),
    resolveStudioNavigationRequest: mocks.resolveStudioNavigationRequest,
    runtime: {},
    sessionId: null,
    skipDecisionRequest: vi.fn(),
    skipInputRequest: vi.fn(),
    studioNavigationRequest,
    studioNavigationStatus: 'pending',
    submitPrompt: vi.fn(),
  }) as unknown as AssistantChatRuntime;

const renderThread = (studioNavigationRequest = makeStudioNavigationRequest()) => {
  const router = createMemoryRouter(
    [
      {
        path: ROUTES.workspace.assistantChat,
        element: <AssistantChatThread chat={makeChat(studioNavigationRequest)} />,
      },
      { path: ROUTES.workspace.guardrails, element: <div data-testid="guardrails-route" /> },
    ],
    { initialEntries: [CHAT_PATH] }
  );

  return render(
    <TestProviders>
      <RouterProvider router={router} />
    </TestProviders>
  );
};

describe('AssistantChatThread Studio UI navigation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('offers the matching Studio UI before continuing with NeMo Assistant', () => {
    renderThread();

    expect(screen.getByText('Studio UI available')).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /1\.\s+Open Guardrails/i })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /2\.\s+Continue in chat/i })).toBeInTheDocument();
    expect(screen.getByTestId('assistant-chat-thread')).toHaveAttribute(
      'data-show-running-indicator',
      'false'
    );
    expect(screen.getByTestId('assistant-chat-thread')).toHaveAttribute(
      'data-animate-message-completion',
      'true'
    );
  });

  it('resolves with navigate and opens the Studio route', async () => {
    const user = userEvent.setup();
    renderThread();

    await user.click(screen.getByRole('option', { name: /1\.\s+Open Guardrails/i }));

    expect(mocks.resolveStudioNavigationRequest).toHaveBeenCalledWith('navigate');
    expect(await screen.findByTestId('guardrails-route')).toBeInTheDocument();
  });

  it('resolves with continue when the user keeps chatting', async () => {
    const user = userEvent.setup();
    renderThread();

    await user.click(screen.getByRole('option', { name: /2\.\s+Continue in chat/i }));

    expect(mocks.resolveStudioNavigationRequest).toHaveBeenCalledWith('continue');
    expect(screen.queryByTestId('guardrails-route')).not.toBeInTheDocument();
  });
});

describe('AssistantChatThread compact mode', () => {
  const renderModeThread = (mode: 'full' | 'compact') =>
    render(
      <TestProviders>
        <RouterProvider
          router={createMemoryRouter(
            [
              {
                path: ROUTES.workspace.assistantChat,
                element: <AssistantChatThread chat={makeChat(null)} mode={mode} />,
              },
            ],
            { initialEntries: [CHAT_PATH] }
          )}
        />
      </TestProviders>
    );

  it('is quiet in the pop-out: greeting, short prompt, two rows, no reset', () => {
    renderModeThread('compact');

    const thread = screen.getByTestId('assistant-chat-thread');
    expect(thread).toHaveAttribute('data-empty-heading', "Hi, I'm Felix");
    expect(thread).toHaveAttribute('data-empty-variant', 'compact');
    expect(thread).toHaveAttribute('data-placeholder', 'Ask Felix…');
    expect(thread).toHaveAttribute('data-min-input-rows', '2');
    expect(thread).toHaveAttribute('data-hide-reset-button', 'true');
  });

  it('keeps the full-size chat as it was', () => {
    renderModeThread('full');

    const thread = screen.getByTestId('assistant-chat-thread');
    expect(thread).toHaveAttribute('data-empty-heading', 'Start a NeMo Assistant session');
    expect(thread).toHaveAttribute('data-empty-variant', 'undefined');
    expect(thread).toHaveAttribute(
      'data-placeholder',
      'Ask NeMo Assistant to work in this workspace'
    );
    expect(thread).toHaveAttribute('data-min-input-rows', '3');
    expect(thread).toHaveAttribute('data-hide-reset-button', 'false');
  });
});
