// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FelixAssistant } from '@studio/components/FelixAssistant';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { FELIX_POSITION_KEY } from '@studio/util/localStorage';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router';

const mocks = vi.hoisted(() => ({
  chat: {
    decisionRequest: null,
    inputRequest: null,
    isRunning: false,
    messages: [],
  } as {
    decisionRequest: unknown;
    inputRequest: unknown;
    isRunning: boolean;
    messages: unknown[];
  },
  startNewChat: vi.fn(),
  // The most recent `enabled` value the component gave the wink hook.
  lastWinkEnabled: undefined as boolean | undefined,
}));

// Wraps the real hook so tests can see whether the component currently allows winking.
vi.mock('@studio/components/FelixAssistant/useFelixWink', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('@studio/components/FelixAssistant/useFelixWink')>();
  return {
    ...actual,
    useFelixWink: (enabled: boolean) => {
      mocks.lastWinkEnabled = enabled;
      return actual.useFelixWink(enabled);
    },
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

const getFelixElement = () => (
  <TestProviders>
    <MemoryRouter initialEntries={['/workspaces/default/jobs']}>
      <FelixAssistant />
    </MemoryRouter>
  </TestProviders>
);

const renderFelix = () => render(getFelixElement());

describe('FelixAssistant', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.chat.isRunning = false;
    mocks.chat.decisionRequest = null;
    mocks.chat.inputRequest = null;
    mocks.chat.messages = [];
  });

  it('renders Felix as a fixed bottom-right trigger with a hover hint', () => {
    renderFelix();

    const trigger = screen.getByRole('button', { name: 'Open NeMo Assistant chat' });
    expect(trigger).toHaveAttribute('title', expect.stringContaining('Ask Felix'));
    expect(trigger).toHaveClass('fixed');
    expect(trigger).toHaveStyle({ right: '16px', bottom: '16px' });
    expect(screen.getByTestId('felix-image')).toBeInTheDocument();
  });

  it('opens and closes the compact chat from Felix', async () => {
    renderFelix();
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Open NeMo Assistant chat' }));
    expect(await screen.findByTestId('compact-chat-thread')).toBeVisible();

    await user.click(screen.getByRole('button', { name: 'Close NeMo Assistant chat' }));
    await waitFor(() => expect(screen.getByTestId('compact-chat-thread')).not.toBeVisible());
  });

  it('anchors the chat to Felix with a bubble tail and does not dim the page', async () => {
    renderFelix();
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Open NeMo Assistant chat' }));

    expect(await screen.findByTestId('felix-bubble-tail')).toBeInTheDocument();
    expect(screen.queryByTestId('assistant-chat-backdrop')).not.toBeInTheDocument();
  });

  it('does not offer a full-page chat from the panel header', async () => {
    renderFelix();
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Open NeMo Assistant chat' }));
    await screen.findByTestId('compact-chat-thread');

    expect(screen.queryByRole('button', { name: 'Open in main chat' })).not.toBeInTheDocument();
  });

  it('starts a new compact chat from the panel header', async () => {
    renderFelix();
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Open NeMo Assistant chat' }));
    // An icon-only button (named for assistive tech), not a labeled "+ New".
    const newChatButton = await screen.findByRole('button', { name: 'New chat' });
    expect(newChatButton).not.toHaveTextContent('New');
    await user.click(newChatButton);

    expect(mocks.startNewChat).toHaveBeenCalledOnce();
  });

  it('closes the compact chat when a chat link is clicked', async () => {
    renderFelix();
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Open NeMo Assistant chat' }));
    expect(await screen.findByTestId('compact-chat-thread')).toBeVisible();

    await user.click(screen.getByRole('link', { name: 'Job details' }));
    await waitFor(() => expect(screen.getByTestId('compact-chat-thread')).not.toBeVisible());
  });

  it('shows a thinking indicator while the agent is running', () => {
    mocks.chat.isRunning = true;

    renderFelix();

    expect(screen.getAllByTestId('assistant-thinking-dot')).toHaveLength(3);
    expect(screen.queryByTestId('assistant-unread-indicator')).not.toBeInTheDocument();
  });

  it('surfaces an unread badge after a response finishes while closed', () => {
    const view = renderFelix();

    expect(screen.queryByTestId('assistant-unread-indicator')).not.toBeInTheDocument();

    mocks.chat.isRunning = true;
    view.rerender(getFelixElement());
    mocks.chat.isRunning = false;
    view.rerender(getFelixElement());

    expect(screen.getByTestId('assistant-unread-indicator')).toBeInTheDocument();
  });

  it('shows the attention badge instead of the thinking dots while awaiting user input', () => {
    mocks.chat.isRunning = true;
    mocks.chat.inputRequest = { requestId: 'request-1', kind: 'dataset_file', input: {} };

    renderFelix();

    expect(screen.getByTestId('assistant-unread-indicator')).toBeInTheDocument();
    expect(screen.queryByTestId('assistant-thinking-indicator')).not.toBeInTheDocument();
  });

  describe('moving Felix', () => {
    // jsdom has no layout: Felix measures 0x0 and the viewport falls back to 1024x768.
    const START = { clientX: 500, clientY: 400 };

    const getTrigger = () => screen.getByTestId('felix-assistant-trigger');
    const getSavedPosition = () => JSON.parse(localStorage.getItem(FELIX_POSITION_KEY) ?? 'null');

    const dragBy = async (
      user: ReturnType<typeof userEvent.setup>,
      deltaX: number,
      deltaY: number
    ) =>
      user.pointer([
        { keys: '[MouseLeft>]', target: getTrigger(), coords: START },
        { coords: { clientX: START.clientX + deltaX, clientY: START.clientY + deltaY } },
        { keys: '[/MouseLeft]' },
      ]);

    beforeEach(() => {
      localStorage.clear();
    });

    it('moves when dragged, remembers the spot, and does not open the chat', async () => {
      renderFelix();
      const user = userEvent.setup();

      // Dragging left/up moves him away from the bottom-right corner.
      await dragBy(user, -100, -50);

      expect(getTrigger()).toHaveStyle({ right: '116px', bottom: '66px' });
      expect(getSavedPosition()).toEqual({ right: 116, bottom: 66 });
      expect(screen.queryByTestId('compact-chat-thread')).not.toBeInTheDocument();
      expect(getTrigger()).toHaveAttribute('aria-label', 'Open NeMo Assistant chat');
    });

    it('treats a tiny movement as a click, not a drag', async () => {
      renderFelix();
      const user = userEvent.setup();

      await dragBy(user, 2, -2);

      expect(await screen.findByTestId('compact-chat-thread')).toBeVisible();
      expect(getSavedPosition()).toBeNull();
      expect(getTrigger()).toHaveStyle({ right: '16px', bottom: '16px' });
    });

    it('can still be clicked open after having been dragged', async () => {
      renderFelix();
      const user = userEvent.setup();

      await dragBy(user, -100, -50);
      await user.click(getTrigger());

      expect(await screen.findByTestId('compact-chat-thread')).toBeVisible();
    });

    it('starts where the user last left him', () => {
      localStorage.setItem(FELIX_POSITION_KEY, JSON.stringify({ right: 300, bottom: 200 }));

      renderFelix();

      expect(getTrigger()).toHaveStyle({ right: '300px', bottom: '200px' });
    });

    it.each([
      ['not an object', '"nope"'],
      ['missing a field', JSON.stringify({ right: 300 })],
      ['non-numeric', JSON.stringify({ right: 'a', bottom: 'b' })],
      ['not valid JSON', '{oops'],
    ])('falls back to the default corner when the saved position is %s', (_label, raw) => {
      localStorage.setItem(FELIX_POSITION_KEY, raw);

      renderFelix();

      expect(getTrigger()).toHaveStyle({ right: '16px', bottom: '16px' });
    });

    it('keeps him fully on-screen when dragged past an edge', async () => {
      renderFelix();
      const user = userEvent.setup();

      // Far down and to the right: pinned to the bottom-right with the edge margin.
      await dragBy(user, 5000, 5000);
      expect(getTrigger()).toHaveStyle({ right: '8px', bottom: '8px' });

      // Far up and to the left: pinned to the top-left, one edge margin in from the viewport.
      await dragBy(user, -5000, -5000);
      expect(getTrigger()).toHaveStyle({ right: '1016px', bottom: '760px' });
    });

    it('returns to the default corner on double-click and forgets the saved spot', async () => {
      localStorage.setItem(FELIX_POSITION_KEY, JSON.stringify({ right: 300, bottom: 200 }));
      renderFelix();
      const user = userEvent.setup();

      await user.dblClick(getTrigger());

      expect(getTrigger()).toHaveStyle({ right: '16px', bottom: '16px' });
      expect(getSavedPosition()).toBeNull();
    });

    it('moves with the arrow keys, and further with Shift', async () => {
      renderFelix();
      const user = userEvent.setup();

      await user.tab();
      expect(getTrigger()).toHaveFocus();

      await user.keyboard('{ArrowLeft}');
      expect(getTrigger()).toHaveStyle({ right: '32px', bottom: '16px' });

      await user.keyboard('{ArrowUp}');
      expect(getTrigger()).toHaveStyle({ right: '32px', bottom: '32px' });

      await user.keyboard('{Shift>}{ArrowUp}{/Shift}');
      expect(getTrigger()).toHaveStyle({ right: '32px', bottom: '96px' });

      // Right/down head back towards the corner, and stop at the edge margin.
      await user.keyboard('{ArrowRight}{ArrowRight}{ArrowRight}');
      expect(getTrigger()).toHaveStyle({ right: '8px' });
      expect(getSavedPosition()).toEqual({ right: 8, bottom: 96 });
    });

    it('opens the panel above him by default, with the tail pointing down at him', async () => {
      renderFelix();
      const user = userEvent.setup();

      await user.click(getTrigger());

      expect(await screen.findByTestId('felix-bubble-tail')).toHaveAttribute('data-side', 'top');
    });

    it('opens the panel below him when he has been moved near the top', async () => {
      localStorage.setItem(FELIX_POSITION_KEY, JSON.stringify({ right: 16, bottom: 700 }));
      renderFelix();
      const user = userEvent.setup();

      await user.click(getTrigger());

      expect(await screen.findByTestId('felix-bubble-tail')).toHaveAttribute('data-side', 'bottom');
    });

    it('closes an open chat when Felix starts being dragged', async () => {
      renderFelix();
      const user = userEvent.setup();

      await user.click(getTrigger());
      expect(await screen.findByTestId('compact-chat-thread')).toBeVisible();
      expect(getTrigger()).toHaveAttribute('aria-label', 'Close NeMo Assistant chat');

      await dragBy(user, -100, -50);

      expect(getTrigger()).toHaveAttribute('aria-label', 'Open NeMo Assistant chat');
      expect(getTrigger()).toHaveStyle({ right: '116px', bottom: '66px' });
    });
  });

  // Real timers on purpose: the lazy-loaded chat thread does not resolve under a faked clock. The
  // wink hook's own tests prove that `enabled=false` stops winking; this checks the component
  // only allows it while idle.
  it('only lets him wink while the chat is closed', async () => {
    const user = userEvent.setup();
    renderFelix();
    expect(mocks.lastWinkEnabled).toBe(true);

    await user.click(screen.getByTestId('felix-assistant-trigger'));
    expect(await screen.findByTestId('compact-chat-thread')).toBeVisible();
    expect(mocks.lastWinkEnabled).toBe(false);

    await user.click(screen.getByRole('button', { name: 'Close NeMo Assistant chat' }));
    await waitFor(() => expect(mocks.lastWinkEnabled).toBe(true));
  });

  describe('winking', () => {
    let randomSpy: ReturnType<typeof vi.spyOn>;

    beforeEach(() => {
      localStorage.clear();
      // Only setTimeout/clearTimeout are faked (that is all the wink uses), so React's scheduler and
      // Testing Library's polling keep running normally. The tests fast-forward explicitly.
      vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
      // 0.5 -> a 30 second gap between winks.
      randomSpy = vi.spyOn(Math, 'random').mockReturnValue(0.5);
    });

    afterEach(() => {
      vi.useRealTimers();
      randomSpy.mockRestore();
    });

    const advance = (ms: number) => {
      act(() => {
        vi.advanceTimersByTime(ms);
      });
    };

    it('shows the open-eyed frame at first, with the wink frame ready underneath', () => {
      renderFelix();

      expect(screen.getByTestId('felix-image')).not.toHaveClass('opacity-0');
      expect(screen.getByTestId('felix-wink-image')).toHaveClass('opacity-0');
      expect(screen.getByTestId('felix-wink-image')).toHaveAttribute('data-winking', 'false');
    });

    it('swaps to the wink frame after about 30 seconds, then back', () => {
      renderFelix();

      advance(30_000);
      expect(screen.getByTestId('felix-wink-image')).toHaveAttribute('data-winking', 'true');
      expect(screen.getByTestId('felix-wink-image')).not.toHaveClass('opacity-0');
      expect(screen.getByTestId('felix-image')).toHaveClass('opacity-0');

      advance(400);
      expect(screen.getByTestId('felix-wink-image')).toHaveAttribute('data-winking', 'false');
      expect(screen.getByTestId('felix-image')).not.toHaveClass('opacity-0');
    });

    it('does not wink while the assistant is working', () => {
      mocks.chat.isRunning = true;
      renderFelix();

      advance(120_000);

      expect(screen.getByTestId('felix-wink-image')).toHaveAttribute('data-winking', 'false');
    });

    it('keeps every frame hidden from assistive technology', () => {
      renderFelix();

      for (const testId of [
        'felix-image',
        'felix-wink-image',
        'felix-frame-hammer-down',
        'felix-frame-hammer-up',
        'felix-frame-book-reading',
        'felix-frame-book-glance',
        'felix-frame-thinking',
        'felix-frame-notepad',
        'felix-frame-done',
      ]) {
        expect(screen.getByTestId(testId)).toHaveAttribute('aria-hidden', 'true');
      }
      expect(screen.getByTestId('felix-assistant-trigger')).toHaveAccessibleName(
        'Open NeMo Assistant chat'
      );
    });
  });

  describe('work poses', () => {
    const assistantCalling = (toolName: string, args: Record<string, string> = {}) => [
      { role: 'user', content: 'do it' },
      {
        role: 'assistant',
        content: [
          {
            type: 'tool-call',
            toolCallId: 'call-1',
            toolName,
            args,
            argsText: JSON.stringify(args),
          },
        ],
      },
    ];

    const getFrame = () => screen.getByTestId('felix-frames').getAttribute('data-frame');
    const advance = (ms: number) => {
      act(() => {
        vi.advanceTimersByTime(ms);
      });
    };

    beforeEach(() => {
      // The pose hooks use setTimeout and setInterval; everything else stays real.
      vi.useFakeTimers({
        toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'],
      });
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('rests in his normal pose when the assistant is idle', () => {
      renderFelix();

      expect(getFrame()).toBe('base');
    });

    it('hammers up and down while the assistant is creating something', () => {
      mocks.chat.isRunning = true;
      mocks.chat.messages = assistantCalling('nemo_api', { action: 'create' });
      renderFelix();

      expect(getFrame()).toBe('hammer-down');
      advance(450);
      expect(getFrame()).toBe('hammer-up');
      advance(450);
      expect(getFrame()).toBe('hammer-down');
    });

    it('shows the matching hammer frame', () => {
      mocks.chat.isRunning = true;
      mocks.chat.messages = assistantCalling('Write');
      renderFelix();

      expect(screen.getByTestId('felix-frame-hammer-down')).not.toHaveClass('opacity-0');
      expect(screen.getByTestId('felix-frame-hammer-up')).toHaveClass('opacity-0');
      expect(screen.getByTestId('felix-image')).toHaveClass('opacity-0');

      advance(450);

      expect(screen.getByTestId('felix-frame-hammer-up')).not.toHaveClass('opacity-0');
      expect(screen.getByTestId('felix-frame-hammer-down')).toHaveClass('opacity-0');
    });

    it('reads the book, eyes on the page, while the assistant looks things up', () => {
      mocks.chat.isRunning = true;
      mocks.chat.messages = assistantCalling('Read');
      renderFelix();

      expect(getFrame()).toBe('book-reading');
      expect(screen.getByTestId('felix-frame-book-reading')).not.toHaveClass('opacity-0');
      expect(screen.getByTestId('felix-frame-book-glance')).toHaveClass('opacity-0');
      // He does not glance up straight away: the shortest wait is 3 seconds.
      advance(2_900);
      expect(getFrame()).toBe('book-reading');
    });

    it('glances up from the book now and then, then goes back to reading', () => {
      // 0 -> the shortest wait between glances, 3 seconds.
      const randomSpy = vi.spyOn(Math, 'random').mockReturnValue(0);
      mocks.chat.isRunning = true;
      mocks.chat.messages = assistantCalling('Read');
      renderFelix();

      advance(2_999);
      expect(getFrame()).toBe('book-reading');
      advance(1);
      expect(getFrame()).toBe('book-glance');
      expect(screen.getByTestId('felix-frame-book-glance')).not.toHaveClass('opacity-0');
      expect(screen.getByTestId('felix-frame-book-reading')).toHaveClass('opacity-0');

      advance(800);
      expect(getFrame()).toBe('book-reading');

      randomSpy.mockRestore();
    });

    it('reads the book for read-only platform lookups too', () => {
      mocks.chat.isRunning = true;
      mocks.chat.messages = assistantCalling('nemo_api', { action: 'list' });
      renderFelix();

      expect(getFrame()).toBe('book-reading');
    });

    it('drops the thinking dots while a pose says it all', () => {
      mocks.chat.isRunning = true;
      mocks.chat.messages = assistantCalling('Read');
      renderFelix();

      expect(screen.queryByTestId('assistant-thinking-indicator')).not.toBeInTheDocument();
    });

    it('keeps the thinking dots when it is working but no pose applies', () => {
      mocks.chat.isRunning = true;
      mocks.chat.messages = assistantCalling('studio_link');
      renderFelix();

      expect(getFrame()).toBe('thinking');
      expect(screen.getAllByTestId('assistant-thinking-dot')).toHaveLength(3);
    });

    it('shows the thinking frame while a run has no tool pose', () => {
      mocks.chat.isRunning = true;
      renderFelix();

      expect(getFrame()).toBe('thinking');
      expect(screen.getByTestId('felix-frame-thinking')).not.toHaveClass('opacity-0');
      expect(screen.getByTestId('felix-image')).toHaveClass('opacity-0');
    });

    it('keeps the idle bob while thinking', () => {
      mocks.chat.isRunning = true;
      renderFelix();

      expect(screen.getByTestId('felix-frames').className).toContain('felix-bob');
    });

    it('lets a tool pose win over the thinking frame', () => {
      mocks.chat.isRunning = true;
      mocks.chat.messages = assistantCalling('Read');
      renderFelix();

      expect(getFrame()).toBe('book-reading');
      expect(screen.getByTestId('felix-frame-thinking')).toHaveClass('opacity-0');
    });

    it('takes notes for 10 seconds when the assistant waits on the user, then rests', () => {
      mocks.chat.isRunning = true;
      mocks.chat.inputRequest = { requestId: 'request-1', kind: 'dataset_file', input: {} };
      renderFelix();

      expect(getFrame()).toBe('notepad');
      expect(screen.getByTestId('felix-frame-notepad')).not.toHaveClass('opacity-0');
      expect(screen.getByTestId('felix-image')).toHaveClass('opacity-0');

      advance(9_999);
      expect(getFrame()).toBe('notepad');
      advance(1);
      // Not thinking either: he is waiting on the user, not working.
      expect(getFrame()).toBe('base');
    });

    it('keeps the attention badge while he takes notes', () => {
      mocks.chat.isRunning = true;
      mocks.chat.inputRequest = { requestId: 'request-1', kind: 'dataset_file', input: {} };
      renderFelix();

      expect(getFrame()).toBe('notepad');
      expect(screen.getByTestId('assistant-unread-indicator')).toBeInTheDocument();
    });

    it('takes notes for a permission request too, and it beats a tool pose', () => {
      mocks.chat.isRunning = true;
      mocks.chat.messages = assistantCalling('Write');
      mocks.chat.decisionRequest = {
        id: 'decision-1',
        title: 'Approval required',
        description: 'Agent wants to use Write.',
      };
      renderFelix();

      expect(getFrame()).toBe('notepad');
    });

    it('leaves the thinking frame once the run ends', () => {
      mocks.chat.isRunning = false;
      renderFelix();

      expect(getFrame()).toBe('base');
    });

    it('does not pose for a stale tool call once the run has ended', () => {
      mocks.chat.isRunning = false;
      mocks.chat.messages = assistantCalling('Write');
      renderFelix();

      expect(getFrame()).toBe('base');
    });

    it('stops the idle bob while posing, so the pose reads cleanly', () => {
      mocks.chat.isRunning = true;
      mocks.chat.messages = assistantCalling('Read');
      renderFelix();

      expect(screen.getByTestId('felix-frames').className).not.toContain('felix-bob');
    });

    it('never winks over a pose', () => {
      mocks.chat.isRunning = true;
      mocks.chat.messages = assistantCalling('Read');
      renderFelix();

      expect(screen.getByTestId('felix-wink-image')).toHaveAttribute('data-winking', 'false');
    });
  });

  describe('celebrating a finished run', () => {
    const reply = (status: Record<string, string>) => [
      { role: 'user', content: 'do it' },
      { role: 'assistant', content: [{ type: 'text', text: 'All done.' }], status },
    ];
    const completed = () => reply({ type: 'complete', reason: 'stop' });

    const getFrame = () => screen.getByTestId('felix-frames').getAttribute('data-frame');
    const advance = (ms: number) => {
      act(() => {
        vi.advanceTimersByTime(ms);
      });
    };

    /** Renders mid-run, then finishes the run with the given reply. */
    const renderRunThatFinishes = (finalMessages: unknown[]) => {
      mocks.chat.isRunning = true;
      mocks.chat.messages = [{ role: 'user', content: 'do it' }];
      const view = renderFelix();

      mocks.chat.isRunning = false;
      mocks.chat.messages = finalMessages;
      view.rerender(getFelixElement());
      return view;
    };

    beforeEach(() => {
      vi.useFakeTimers({
        toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'],
      });
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('throws his hands up for 3 seconds when a run finishes, then goes back to normal', () => {
      renderRunThatFinishes(completed());

      expect(getFrame()).toBe('done');
      expect(screen.getByTestId('felix-frame-done')).not.toHaveClass('opacity-0');
      expect(screen.getByTestId('felix-image')).toHaveClass('opacity-0');

      advance(2_999);
      expect(getFrame()).toBe('done');
      advance(1);
      expect(getFrame()).toBe('base');
    });

    it('hops while he celebrates, instead of the idle bob', () => {
      renderRunThatFinishes(completed());

      const frames = screen.getByTestId('felix-frames');
      expect(frames.className).toContain('felix-hop');
      expect(frames.className).not.toContain('felix-bob');

      advance(3_000);

      expect(frames.className).not.toContain('felix-hop');
      expect(frames.className).toContain('felix-bob');
    });

    it('does not celebrate a run that failed', () => {
      renderRunThatFinishes(reply({ type: 'incomplete', reason: 'error', error: 'boom' }));

      expect(getFrame()).toBe('base');
    });

    it('does not celebrate a run the user stopped', () => {
      renderRunThatFinishes(reply({ type: 'incomplete', reason: 'cancelled' }));

      expect(getFrame()).toBe('base');
    });

    it('does not celebrate a conversation that was already finished', () => {
      mocks.chat.isRunning = false;
      mocks.chat.messages = completed();
      renderFelix();

      expect(getFrame()).toBe('base');
    });

    it('stops celebrating and gets back to work when a new run starts', () => {
      const view = renderRunThatFinishes(completed());
      expect(getFrame()).toBe('done');

      mocks.chat.isRunning = true;
      mocks.chat.messages = [
        ...completed(),
        { role: 'user', content: 'one more thing' },
        {
          role: 'assistant',
          content: [
            { type: 'tool-call', toolCallId: 'c1', toolName: 'Write', args: {}, argsText: '{}' },
          ],
        },
      ];
      view.rerender(getFelixElement());

      expect(getFrame()).toBe('hammer-down');
    });

    it('does not wink while celebrating', () => {
      renderRunThatFinishes(completed());

      expect(screen.getByTestId('felix-wink-image')).toHaveAttribute('data-winking', 'false');
    });
  });
});
