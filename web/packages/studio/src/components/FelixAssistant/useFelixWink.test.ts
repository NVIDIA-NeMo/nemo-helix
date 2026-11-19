// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getWinkDelayMs,
  useFelixWink,
  WINK_HOLD_MS,
  WINK_MAX_DELAY_MS,
  WINK_MIN_DELAY_MS,
} from '@studio/components/FelixAssistant/useFelixWink';
import { act, renderHook } from '@testing-library/react';

const originalMatchMedia = window.matchMedia;

const mockReducedMotion = (matches: boolean) => {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches,
    media: query,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  }));
};

const setPageVisibility = (state: 'visible' | 'hidden') => {
  Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => state });
  act(() => {
    document.dispatchEvent(new Event('visibilitychange'));
  });
};

const advance = (ms: number) => {
  act(() => {
    vi.advanceTimersByTime(ms);
  });
};

describe('getWinkDelayMs', () => {
  it('stays within 25-35 seconds', () => {
    expect(getWinkDelayMs(() => 0)).toBe(WINK_MIN_DELAY_MS);
    expect(getWinkDelayMs(() => 0.5)).toBe(30_000);
    expect(getWinkDelayMs(() => 1)).toBe(WINK_MAX_DELAY_MS);
  });
});

describe('useFelixWink', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    // A fixed 30s gap keeps the timeline readable.
    vi.spyOn(Math, 'random').mockReturnValue(0.5);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    window.matchMedia = originalMatchMedia;
    setPageVisibility('visible');
  });

  it('does not wink at first', () => {
    const { result } = renderHook(() => useFelixWink(true));

    expect(result.current).toBe(false);
    advance(29_999);
    expect(result.current).toBe(false);
  });

  it('winks after the delay, holds briefly, then waits for the next one', () => {
    const { result } = renderHook(() => useFelixWink(true));

    advance(30_000);
    expect(result.current).toBe(true);

    advance(WINK_HOLD_MS - 1);
    expect(result.current).toBe(true);
    advance(1);
    expect(result.current).toBe(false);

    // And again, a full delay later.
    advance(29_999);
    expect(result.current).toBe(false);
    advance(1);
    expect(result.current).toBe(true);
  });

  it('never winks while he is busy', () => {
    const { result } = renderHook(() => useFelixWink(false));

    advance(120_000);

    expect(result.current).toBe(false);
  });

  it('ends a wink in progress and cancels the next one when he becomes busy', () => {
    const { result, rerender } = renderHook(({ enabled }) => useFelixWink(enabled), {
      initialProps: { enabled: true },
    });

    advance(30_000);
    expect(result.current).toBe(true);

    rerender({ enabled: false });
    expect(result.current).toBe(false);

    advance(120_000);
    expect(result.current).toBe(false);
  });

  it('starts a fresh delay when he becomes idle again', () => {
    const { result, rerender } = renderHook(({ enabled }) => useFelixWink(enabled), {
      initialProps: { enabled: false },
    });
    advance(100_000);

    rerender({ enabled: true });
    advance(29_999);
    expect(result.current).toBe(false);
    advance(1);
    expect(result.current).toBe(true);
  });

  it('does not wink for users who prefer reduced motion', () => {
    mockReducedMotion(true);
    const { result } = renderHook(() => useFelixWink(true));

    advance(120_000);

    expect(result.current).toBe(false);
  });

  it('winks normally when reduced motion is not requested', () => {
    mockReducedMotion(false);
    const { result } = renderHook(() => useFelixWink(true));

    advance(30_000);

    expect(result.current).toBe(true);
  });

  it('pauses while the tab is hidden and resumes with a fresh delay', () => {
    const { result } = renderHook(() => useFelixWink(true));

    advance(10_000);
    setPageVisibility('hidden');
    advance(120_000);
    expect(result.current).toBe(false);

    setPageVisibility('visible');
    advance(29_999);
    expect(result.current).toBe(false);
    advance(1);
    expect(result.current).toBe(true);
  });
});
