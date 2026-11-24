// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  CELEBRATION_MS,
  useFelixCelebration,
} from '@studio/components/FelixAssistant/useFelixCelebration';
import { act, renderHook } from '@testing-library/react';

interface Props {
  isRunning: boolean;
  lastReplySucceeded: boolean;
}

const renderCelebration = (initialProps: Props) =>
  renderHook((props: Props) => useFelixCelebration(props), { initialProps });

const advance = (ms: number) => {
  act(() => {
    vi.advanceTimersByTime(ms);
  });
};

describe('useFelixCelebration', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('does not celebrate at first', () => {
    const { result } = renderCelebration({ isRunning: false, lastReplySucceeded: false });

    expect(result.current).toBe(false);
  });

  it('celebrates for 3 seconds when a run finishes successfully, then stops', () => {
    const { result, rerender } = renderCelebration({ isRunning: true, lastReplySucceeded: false });

    rerender({ isRunning: false, lastReplySucceeded: true });
    expect(result.current).toBe(true);

    advance(CELEBRATION_MS - 1);
    expect(result.current).toBe(true);
    advance(1);
    expect(result.current).toBe(false);
    expect(CELEBRATION_MS).toBe(3_000);
  });

  it('does not celebrate a run that failed or was cancelled', () => {
    const { result, rerender } = renderCelebration({ isRunning: true, lastReplySucceeded: false });

    rerender({ isRunning: false, lastReplySucceeded: false });

    expect(result.current).toBe(false);
  });

  it('does not celebrate just because an old finished conversation was loaded', () => {
    // Not running throughout; only the reply changes (e.g. switching to a finished session).
    const { result, rerender } = renderCelebration({ isRunning: false, lastReplySucceeded: false });

    rerender({ isRunning: false, lastReplySucceeded: true });

    expect(result.current).toBe(false);
  });

  it('does not celebrate when mounted with a finished conversation', () => {
    const { result } = renderCelebration({ isRunning: false, lastReplySucceeded: true });

    expect(result.current).toBe(false);
  });

  it('ends the celebration straight away when a new run starts', () => {
    const { result, rerender } = renderCelebration({ isRunning: true, lastReplySucceeded: false });
    rerender({ isRunning: false, lastReplySucceeded: true });
    expect(result.current).toBe(true);

    rerender({ isRunning: true, lastReplySucceeded: false });

    expect(result.current).toBe(false);
    // And the old timer must not leave anything behind.
    advance(CELEBRATION_MS * 2);
    expect(result.current).toBe(false);
  });

  it('celebrates again after the next successful run', () => {
    const { result, rerender } = renderCelebration({ isRunning: true, lastReplySucceeded: false });
    rerender({ isRunning: false, lastReplySucceeded: true });
    advance(CELEBRATION_MS);
    expect(result.current).toBe(false);

    rerender({ isRunning: true, lastReplySucceeded: false });
    rerender({ isRunning: false, lastReplySucceeded: true });

    expect(result.current).toBe(true);
  });
});
