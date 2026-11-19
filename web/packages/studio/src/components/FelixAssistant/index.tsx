// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Add, Close } from '@nemo/icons/icons';
import {
  Button,
  Flex,
  Popover,
  Spinner,
  Stack,
  Text,
  Tooltip,
} from '@nvidia/foundations-react-core';
import felixBookGlanceImage from '@studio/assets/felix-book-glance.png';
import felixBookReadingImage from '@studio/assets/felix-book-reading.png';
import felixDoneImage from '@studio/assets/felix-done.png';
import felixHammerDownImage from '@studio/assets/felix-hammer-down.png';
import felixHammerUpImage from '@studio/assets/felix-hammer-up.png';
import felixWinkImage from '@studio/assets/felix-wink.png';
import felixImage from '@studio/assets/felix.png';
import {
  didLastReplySucceed,
  getFelixActivity,
} from '@studio/components/FelixAssistant/felixActivity';
import { useBubbleTail } from '@studio/components/FelixAssistant/useBubbleTail';
import { useFelixCelebration } from '@studio/components/FelixAssistant/useFelixCelebration';
import { useFelixPose } from '@studio/components/FelixAssistant/useFelixPose';
import { useFelixPosition } from '@studio/components/FelixAssistant/useFelixPosition';
import { useFelixWink } from '@studio/components/FelixAssistant/useFelixWink';
import { ChatThreadErrorBoundary } from '@studio/routes/agents/AssistantChatRoute/ChatThreadErrorBoundary';
import { useAssistantChatContext } from '@studio/routes/agents/AssistantChatRoute/context/useAssistantChatContext';
import {
  lazy,
  Suspense,
  type FC,
  type MouseEvent,
  type PointerEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

// Static import would pull the whole chat surface into the entry chunk, since
// Felix renders on every workspace page.
const importChatThread = () =>
  import('@studio/routes/agents/AssistantChatRoute/AssistantChatThread');

// lazy() caches a rejected import forever, so a retry needs a fresh component.
const createChatThread = () =>
  lazy(() => importChatThread().then((m) => ({ default: m.AssistantChatThread })));

const preloadChatThread = () => void importChatThread().catch(() => undefined);

const OPEN_LABEL = 'Open NeMo Assistant chat';
const CLOSE_LABEL = 'Close NeMo Assistant chat';
const HOVER_HINT = 'Ask Felix (drag to move, double-click to reset)';

type FelixFrame =
  | 'base'
  | 'wink'
  | 'hammer-down'
  | 'hammer-up'
  | 'book-reading'
  | 'book-glance'
  | 'done';

const FRAME_SOURCES: Record<FelixFrame, string> = {
  base: felixImage,
  wink: felixWinkImage,
  'hammer-down': felixHammerDownImage,
  'hammer-up': felixHammerUpImage,
  'book-reading': felixBookReadingImage,
  'book-glance': felixBookGlanceImage,
  done: felixDoneImage,
};

/** Stacked on top of the base frame, which sets the size. */
const OVERLAY_FRAMES: readonly Exclude<FelixFrame, 'base'>[] = [
  'wink',
  'hammer-down',
  'hammer-up',
  'book-reading',
  'book-glance',
  'done',
];

/**
 * Which frame to show. Celebrating a finished run wins over everything; then a work pose; then a
 * wink; then the resting pose.
 */
const getFelixFrame = ({
  isCelebrating,
  pose,
  hammerUp,
  glancingUp,
  isWinking,
}: {
  isCelebrating: boolean;
  pose: 'building' | 'docs' | null;
  hammerUp: boolean;
  glancingUp: boolean;
  isWinking: boolean;
}): FelixFrame => {
  if (isCelebrating) return 'done';
  if (pose === 'building') return hammerUp ? 'hammer-up' : 'hammer-down';
  // Eyes on the page, with an occasional glance up.
  if (pose === 'docs') return glancingUp ? 'book-glance' : 'book-reading';
  return isWinking ? 'wink' : 'base';
};

const chatThreadFallback = (
  <Flex align="center" justify="center" className="h-full">
    <Spinner size="medium" aria-label="Loading chat..." />
  </Flex>
);

/**
 * Felix, the NeMo Assistant mascot: a character that starts in the bottom-right of the screen and
 * opens the assistant chat in a panel anchored to him. The user can drag him (or use the arrow
 * keys) to reposition him; the spot is remembered and the panel and its tail follow him.
 *
 * This is a thin view of the shared chat runtime (owned by AssistantChatProvider). Because the
 * runtime lives above the routes, opening the panel mid-run shows the live stream and
 * thinking/awaiting-input state. It must render inside that provider.
 */
export const FelixAssistant: FC = () => {
  const { chat, startNewChat } = useAssistantChatContext();
  const { decisionRequest, inputRequest, isRunning } = chat;
  const [isOpen, setIsOpen] = useState(false);
  const [hasUnreadResponse, setHasUnreadResponse] = useState(false);
  const [scrollToBottomSignal, setScrollToBottomSignal] = useState(0);
  const [hasOpened, setHasOpened] = useState(false);
  const [ChatThread, setChatThread] = useState(createChatThread);

  // Starting to drag Felix closes the panel so it isn't left floating where he used to be.
  const closeChat = useCallback(() => setIsOpen(false), []);
  const {
    ref: triggerRef,
    position,
    placement,
    isDragging,
    consumeDragClick,
    handlers: dragHandlers,
  } = useFelixPosition({ onDragStart: closeChat });
  const contentRef = useRef<HTMLDivElement>(null);
  const tailLeft = useBubbleTail({ isOpen, anchorRef: triggerRef, contentRef });
  // For a few seconds after a run finishes successfully he cheers, hands up (even with the chat
  // open: it is his reaction to the work being done).
  const isCelebrating = useFelixCelebration({
    isRunning,
    lastReplySucceeded: didLastReplySucceed(chat.messages),
  });
  // He only winks while idle: chat closed, nothing running, not celebrating, and not being moved.
  const isWinking = useFelixWink(!isOpen && !isDragging && !isRunning && !isCelebrating);
  // What the assistant is doing decides his pose: hammering while it builds, reading while it
  // looks things up. The pose follows the work even with the chat open.
  const activity = useMemo(() => getFelixActivity(chat.messages), [chat.messages]);
  const { pose, hammerUp, glancingUp } = useFelixPose({ activity, isRunning });
  const frame = getFelixFrame({ isCelebrating, pose, hammerUp, glancingUp, isWinking });

  const retryChatThread = useCallback(() => setChatThread(() => createChatThread()), []);

  // While the agent is blocked on a permission/input request the stream stays
  // open (isRunning is still true), but it is waiting on the user rather than
  // thinking — surface the attention badge instead of the loading dots.
  const isAwaitingUserInput = !!decisionRequest || !!inputRequest;

  // Surface an unread badge when the agent finishes responding while the
  // panel is closed, and clear it the moment the user opens the chat.
  const wasRunningRef = useRef(false);
  useEffect(() => {
    if (isOpen) {
      setHasUnreadResponse(false);
    } else if (wasRunningRef.current && !isRunning) {
      setHasUnreadResponse(true);
    }
    wasRunningRef.current = isRunning;
  }, [isOpen, isRunning]);

  // The popover dismisses on Esc / outside click; just mirror that here.
  const handleOpenChange = useCallback((open: boolean) => {
    if (!open) setIsOpen(false);
  }, []);

  // The controlled popover treats this trigger as "outside" its content, so a
  // pointer-down would dismiss it right before the click re-opens it. Stop that
  // dismissal while open and let the click own the toggle.
  const handleTriggerPointerDown = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      if (isOpen) event.stopPropagation();
    },
    [isOpen]
  );

  const handleTriggerPointerDownWithDrag = useCallback(
    (event: PointerEvent<HTMLButtonElement>) => {
      handleTriggerPointerDown(event);
      dragHandlers.onPointerDown(event);
    },
    [dragHandlers, handleTriggerPointerDown]
  );

  const handleTriggerClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      // The click that ends a drag is not a request to open or close the chat.
      if (consumeDragClick()) return;
      if (isOpen) {
        setIsOpen(false);
        return;
      }
      setIsOpen(true);
      setHasOpened(true);
      setScrollToBottomSignal((signal) => signal + 1);
    },
    [consumeDragClick, isOpen]
  );

  // Close the panel when the user follows an in-chat link to another route.
  // This runs on the bubble phase (not capture) so the link's own react-router
  // navigation happens first — closing the popover out from under the anchor
  // mid-click breaks client-side navigation and triggers a full reload.
  const handlePopoverLinkClick = useCallback((event: MouseEvent<HTMLElement>) => {
    if (event.target instanceof Element && event.target.closest('a[href]')) {
      setIsOpen(false);
    }
  }, []);

  return (
    <Popover
      ref={contentRef}
      align="end"
      open={isOpen}
      side={placement.side}
      className="relative z-[1001] data-[side=bottom]:mt-3.5 data-[side=top]:mb-3.5 !max-h-[calc(100vh-16px)] max-w-[calc(100vw-32px)] !overflow-visible !rounded-xl !bg-surface-sunken !p-0 dark:!bg-surface-raised"
      onOpenChange={handleOpenChange}
      slotContent={
        <>
          <Stack
            // eslint-disable-next-line no-restricted-syntax -- the panel's height fits the room left around Felix, measured at runtime
            style={{ height: placement.maxHeight }}
            className="min-h-0 w-[min(420px,calc(100vw-32px))] overflow-hidden rounded-xl bg-surface-sunken p-density-md text-primary dark:bg-surface-raised"
            gap="density-md"
            onClick={handlePopoverLinkClick}
          >
            <Flex align="center" justify="between" gap="density-md">
              <Text kind="label/bold/md">Felix</Text>
              <Flex align="center" gap="density-xs">
                <Tooltip slotContent="New chat" side="bottom">
                  <Button kind="tertiary" size="small" aria-label="New chat" onClick={startNewChat}>
                    <Add size="16" />
                  </Button>
                </Tooltip>
                <Button
                  kind="tertiary"
                  size="small"
                  aria-label="Close chat window"
                  onClick={() => setIsOpen(false)}
                >
                  <Close size="16" />
                </Button>
              </Flex>
            </Flex>
            <Stack className="min-h-0 flex-1 overflow-hidden">
              {hasOpened ? (
                <ChatThreadErrorBoundary onRetry={retryChatThread}>
                  <Suspense fallback={chatThreadFallback}>
                    <ChatThread
                      chat={chat}
                      mode="compact"
                      scrollToBottomSignal={scrollToBottomSignal}
                    />
                  </Suspense>
                </ChatThreadErrorBoundary>
              ) : null}
            </Stack>
          </Stack>
          {/* Speech-bubble tail pointing at Felix, wherever the panel ended up relative to him. */}
          {tailLeft !== null ? (
            <span
              aria-hidden="true"
              data-testid="felix-bubble-tail"
              data-side={placement.side}
              // eslint-disable-next-line no-restricted-syntax -- the tail's offset is measured at runtime to point at Felix
              style={{ left: tailLeft }}
              className={`pointer-events-none absolute size-4 rotate-45 border-base bg-surface-sunken dark:bg-surface-raised ${
                placement.side === 'top'
                  ? '-bottom-2 border-b border-r'
                  : '-top-2 border-l border-t'
              }`}
            />
          ) : null}
        </>
      }
    >
      <button
        ref={triggerRef}
        type="button"
        aria-label={isOpen ? CLOSE_LABEL : OPEN_LABEL}
        title={isOpen ? CLOSE_LABEL : HOVER_HINT}
        data-testid="felix-assistant-trigger"
        data-dragging={isDragging}
        // eslint-disable-next-line no-restricted-syntax -- Felix's position is a user-dragged pixel value that cannot be a class
        style={{ right: position.right, bottom: position.bottom }}
        // touch-none: stop the browser scrolling the page instead of letting touch drag Felix.
        className={`fixed z-[1000] touch-none rounded-xl p-1 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand ${
          isDragging ? 'cursor-grabbing' : 'cursor-grab'
        }`}
        onMouseEnter={preloadChatThread}
        onPointerDown={handleTriggerPointerDownWithDrag}
        onPointerMove={dragHandlers.onPointerMove}
        onPointerUp={dragHandlers.onPointerUp}
        onPointerCancel={dragHandlers.onPointerCancel}
        onKeyDown={dragHandlers.onKeyDown}
        onDoubleClick={dragHandlers.onDoubleClick}
        onClick={handleTriggerClick}
      >
        {/* Every frame lives in the DOM from the start (so a pose never waits on a download) and
            shares this wrapper so the bob and hover scale move them together. Only one shows at a
            time; stacking them would darken the translucent helmet glass. They are all the same
            size and cropped to the same box, so swapping never shifts the character. */}
        <span
          data-testid="felix-frames"
          data-frame={frame}
          className={`relative block transition-transform ${
            isDragging ? 'scale-110' : 'hover:scale-105'
          } ${
            isCelebrating
              ? // Five 0.6s hops fill the 3 second celebration.
                'motion-safe:animate-[felix-hop_0.6s_ease-in-out_5]'
              : isOpen || isDragging || pose
                ? ''
                : 'motion-safe:animate-[felix-bob_3.2s_ease-in-out_infinite]'
          }`}
        >
          <img
            src={FRAME_SOURCES.base}
            alt=""
            aria-hidden="true"
            data-testid="felix-image"
            draggable={false}
            className={`block h-24 w-auto select-none drop-shadow-lg ${
              frame === 'base' ? '' : 'opacity-0'
            }`}
          />
          {OVERLAY_FRAMES.map((overlayFrame) => (
            <img
              key={overlayFrame}
              src={FRAME_SOURCES[overlayFrame]}
              alt=""
              aria-hidden="true"
              data-testid={
                overlayFrame === 'wink' ? 'felix-wink-image' : `felix-frame-${overlayFrame}`
              }
              data-winking={overlayFrame === 'wink' ? frame === 'wink' : undefined}
              draggable={false}
              className={`absolute inset-0 size-full select-none drop-shadow-lg ${
                frame === overlayFrame ? '' : 'opacity-0'
              }`}
            />
          ))}
        </span>
        {/* The hammer or book already says "working", so the dots only show with no pose. */}
        {isRunning && !isAwaitingUserInput && !pose ? (
          <span
            className="pointer-events-none absolute right-1 top-1 flex h-4 w-6 items-center justify-center gap-0.5 rounded-full border border-base bg-surface-sunken/90 dark:bg-surface-raised/90"
            data-testid="assistant-thinking-indicator"
          >
            <span
              className="size-1 rounded-full bg-brand motion-safe:animate-pulse"
              data-testid="assistant-thinking-dot"
            />
            <span
              className="size-1 rounded-full bg-brand motion-safe:animate-pulse [animation-delay:150ms]"
              data-testid="assistant-thinking-dot"
            />
            <span
              className="size-1 rounded-full bg-brand motion-safe:animate-pulse [animation-delay:300ms]"
              data-testid="assistant-thinking-dot"
            />
          </span>
        ) : isAwaitingUserInput || hasUnreadResponse ? (
          <span
            className="pointer-events-none absolute right-2 top-2 size-3 rounded-full bg-brand ring-2 ring-surface-sunken dark:ring-surface-raised"
            data-testid="assistant-unread-indicator"
          />
        ) : null}
      </button>
    </Popover>
  );
};
