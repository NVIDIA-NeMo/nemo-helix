// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type RefObject, useCallback, useEffect, useRef } from 'react';

interface UseStickToBottomOptions {
  /** When false, the scroll listener and observer are detached (e.g. while loading). */
  enabled?: boolean;
  /** Distance (px) from the bottom that still counts as "at the bottom". */
  threshold?: number;
  /**
   * Changing this re-attaches the observers and re-arms auto-scroll — use it when the
   * scroll container's content is swapped out (e.g. toggling "show all" vs "tail").
   */
  resetKey?: unknown;
}

interface UseStickToBottom<T extends HTMLElement> {
  /** Attach to the scrollable element (or the element whose content grows). */
  ref: RefObject<T | null>;
  /** Jump to the bottom now and re-arm auto-scroll so future growth stays pinned. */
  scrollToBottom: () => void;
}

/**
 * Keeps a scroll container pinned to the bottom as content streams in, but only while
 * the user is already at the bottom. If the user scrolls up, auto-scroll pauses until
 * they scroll back down (within `threshold`).
 *
 * Growth is detected with a MutationObserver so it also catches async content (e.g. a
 * CodeSnippet that re-renders highlighted text after the value prop changes).
 */
export function useStickToBottom<T extends HTMLElement = HTMLElement>({
  enabled = true,
  threshold = 50,
  resetKey,
}: UseStickToBottomOptions = {}): UseStickToBottom<T> {
  const ref = useRef<T>(null);
  const shouldAutoScrollRef = useRef(true);

  const scrollToBottom = useCallback(() => {
    shouldAutoScrollRef.current = true;
    const element = ref.current;
    if (element) {
      element.scrollTop = element.scrollHeight - element.clientHeight;
    }
  }, []);

  // Pin to the bottom whenever content changes and the user is at the bottom.
  useEffect(() => {
    if (!enabled) return;
    const element = ref.current;
    if (!element) return;

    shouldAutoScrollRef.current = true;
    element.scrollTop = element.scrollHeight - element.clientHeight;

    const observer = new MutationObserver(() => {
      if (shouldAutoScrollRef.current) {
        element.scrollTop = element.scrollHeight - element.clientHeight;
      }
    });

    observer.observe(element, { childList: true, subtree: true, characterData: true });

    return () => observer.disconnect();
  }, [enabled, resetKey]);

  // Track whether the user is at the bottom so we can pause/resume auto-scroll.
  useEffect(() => {
    if (!enabled) return;
    const element = ref.current;
    if (!element) return;

    const handleScroll = () => {
      const distanceFromBottom = element.scrollHeight - element.clientHeight - element.scrollTop;
      shouldAutoScrollRef.current = Math.abs(distanceFromBottom) < threshold;
    };

    element.addEventListener('scroll', handleScroll);

    return () => element.removeEventListener('scroll', handleScroll);
  }, [enabled, threshold, resetKey]);

  return { ref, scrollToBottom };
}
