// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { usePrefersReducedMotion } from '@studio/components/FelixAssistant/usePrefersReducedMotion';
import { useEffect, useState } from 'react';

/** Felix winks every 25-35 seconds; the gap is randomised so it never feels like clockwork. */
export const WINK_MIN_DELAY_MS = 25_000;
export const WINK_MAX_DELAY_MS = 35_000;
/** How long the wink frame is held: long enough to read as a wink, short enough to stay a blink. */
export const WINK_HOLD_MS = 400;

export const getWinkDelayMs = (random: () => number = Math.random): number =>
  WINK_MIN_DELAY_MS + random() * (WINK_MAX_DELAY_MS - WINK_MIN_DELAY_MS);

const useIsPageVisible = (): boolean => {
  const [visible, setVisible] = useState(() => document.visibilityState !== 'hidden');

  useEffect(() => {
    const handleChange = () => setVisible(document.visibilityState !== 'hidden');
    document.addEventListener('visibilitychange', handleChange);
    return () => document.removeEventListener('visibilitychange', handleChange);
  }, []);

  return visible;
};

/**
 * Tells Felix when to wink: briefly, at random intervals, while he is idle.
 *
 * - `enabled` is the caller's idle check (e.g. chat closed, not mid-run, not being dragged). Losing
 *   it cancels any pending wink and ends one in progress.
 * - No winking for users who prefer reduced motion, and none while the tab is hidden; both resume
 *   with a fresh random delay rather than firing a backlog.
 */
export const useFelixWink = (enabled: boolean): boolean => {
  const [isWinking, setIsWinking] = useState(false);
  const prefersReducedMotion = usePrefersReducedMotion();
  const isPageVisible = useIsPageVisible();
  const canWink = enabled && !prefersReducedMotion && isPageVisible;

  useEffect(() => {
    if (!canWink) {
      setIsWinking(false);
      return;
    }

    let timer: ReturnType<typeof setTimeout>;
    const scheduleWink = () => {
      timer = setTimeout(() => {
        setIsWinking(true);
        timer = setTimeout(() => {
          setIsWinking(false);
          scheduleWink();
        }, WINK_HOLD_MS);
      }, getWinkDelayMs());
    };
    scheduleWink();

    return () => clearTimeout(timer);
  }, [canWink]);

  return isWinking;
};
