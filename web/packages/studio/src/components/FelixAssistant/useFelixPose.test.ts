// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FelixActivity } from '@studio/components/FelixAssistant/felixActivity';
import {
  GLANCE_HOLD_MS,
  GLANCE_MAX_DELAY_MS,
  GLANCE_MIN_DELAY_MS,
  getGlanceDelayMs,
  HAMMER_FRAME_MS,
  POSE_HOLD_MS,
  useFelixPose,
} from '@studio/components/FelixAssistant/useFelixPose';
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

const advance = (ms: number) => {
  act(() => {
    vi.advanceTimersByTime(ms);
  });
};

interface Props {
  activity: FelixActivity | null;
  isRunning: boolean;
}

const renderPose = (initialProps: Props) =>
  renderHook((props: Props) => useFelixPose(props), { initialProps });

describe('useFelixPose', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    window.matchMedia = originalMatchMedia;
  });

  it('has no pose when nothing matching is happening', () => {
    const { result } = renderPose({ activity: null, isRunning: true });

    expect(result.current.pose).toBeNull();
    expect(result.current.hammerUp).toBe(false);
  });

  it('shows a pose immediately when work starts', () => {
    const { result, rerender } = renderPose({ activity: null, isRunning: true });

    rerender({ activity: 'docs', isRunning: true });
    expect(result.current.pose).toBe('docs');

    rerender({ activity: 'building', isRunning: true });
    expect(result.current.pose).toBe('building');
  });

  it('holds the pose briefly after the tool call ends, then lets it go', () => {
    const { result, rerender } = renderPose({ activity: 'docs', isRunning: true });

    rerender({ activity: null, isRunning: true });
    advance(POSE_HOLD_MS - 1);
    expect(result.current.pose).toBe('docs');

    advance(1);
    expect(result.current.pose).toBeNull();
  });

  it('keeps a steady pose through quick back-to-back tool calls', () => {
    const { result, rerender } = renderPose({ activity: 'building', isRunning: true });

    rerender({ activity: null, isRunning: true });
    advance(POSE_HOLD_MS - 500);
    rerender({ activity: 'building', isRunning: true });
    advance(POSE_HOLD_MS * 2);

    // The gap was shorter than the hold, so the pose never dropped; with a call in flight it stays.
    expect(result.current.pose).toBe('building');
  });

  it('ends the pose straight away when the run ends', () => {
    const { result, rerender } = renderPose({ activity: 'building', isRunning: true });

    rerender({ activity: 'building', isRunning: false });

    expect(result.current.pose).toBeNull();
  });

  it('swings the hammer up and down while building', () => {
    const { result } = renderPose({ activity: 'building', isRunning: true });
    expect(result.current.hammerUp).toBe(false);

    advance(HAMMER_FRAME_MS);
    expect(result.current.hammerUp).toBe(true);

    advance(HAMMER_FRAME_MS);
    expect(result.current.hammerUp).toBe(false);

    advance(HAMMER_FRAME_MS);
    expect(result.current.hammerUp).toBe(true);
  });

  it('puts the hammer back at rest when he stops building', () => {
    const { result, rerender } = renderPose({ activity: 'building', isRunning: true });
    advance(HAMMER_FRAME_MS);
    expect(result.current.hammerUp).toBe(true);

    rerender({ activity: 'docs', isRunning: true });

    expect(result.current.hammerUp).toBe(false);
    advance(HAMMER_FRAME_MS * 4);
    expect(result.current.hammerUp).toBe(false);
  });

  it('does not swing for anything but building', () => {
    const { result } = renderPose({ activity: 'docs', isRunning: true });

    advance(HAMMER_FRAME_MS * 4);

    expect(result.current.hammerUp).toBe(false);
  });

  it('keeps the hammer still for users who prefer reduced motion, but still builds', () => {
    mockReducedMotion(true);
    const { result } = renderPose({ activity: 'building', isRunning: true });

    advance(HAMMER_FRAME_MS * 6);

    expect(result.current.pose).toBe('building');
    expect(result.current.hammerUp).toBe(false);
  });

  describe('glancing up from the book', () => {
    let randomSpy: ReturnType<typeof vi.spyOn>;

    beforeEach(() => {
      // 0.5 -> 4.5 seconds between glances, the middle of the 3-6 second range.
      randomSpy = vi.spyOn(Math, 'random').mockReturnValue(0.5);
    });

    afterEach(() => {
      randomSpy.mockRestore();
    });

    it('keeps the wait between glances within 3-6 seconds', () => {
      expect(getGlanceDelayMs(() => 0)).toBe(GLANCE_MIN_DELAY_MS);
      expect(getGlanceDelayMs(() => 0.5)).toBe(4_500);
      expect(getGlanceDelayMs(() => 1)).toBe(GLANCE_MAX_DELAY_MS);
    });

    it('starts with his eyes on the page', () => {
      const { result } = renderPose({ activity: 'docs', isRunning: true });

      expect(result.current.glancingUp).toBe(false);
      advance(4_499);
      expect(result.current.glancingUp).toBe(false);
    });

    it('glances up after the wait, holds briefly, then goes back to the page', () => {
      const { result } = renderPose({ activity: 'docs', isRunning: true });

      advance(4_500);
      expect(result.current.glancingUp).toBe(true);

      advance(GLANCE_HOLD_MS - 1);
      expect(result.current.glancingUp).toBe(true);
      advance(1);
      expect(result.current.glancingUp).toBe(false);
    });

    it('keeps glancing now and then for as long as he reads', () => {
      const { result } = renderPose({ activity: 'docs', isRunning: true });

      advance(4_500 + GLANCE_HOLD_MS);
      expect(result.current.glancingUp).toBe(false);
      advance(4_500);
      expect(result.current.glancingUp).toBe(true);
    });

    it('stops glancing when he stops reading', () => {
      const { result, rerender } = renderPose({ activity: 'docs', isRunning: true });
      advance(4_500);
      expect(result.current.glancingUp).toBe(true);

      rerender({ activity: 'building', isRunning: true });

      expect(result.current.glancingUp).toBe(false);
      advance(60_000);
      expect(result.current.glancingUp).toBe(false);
    });

    it('starts a fresh wait with each new reading spell', () => {
      const { result, rerender } = renderPose({ activity: 'docs', isRunning: true });
      advance(4_000);
      rerender({ activity: 'building', isRunning: true });
      rerender({ activity: 'docs', isRunning: true });

      // 4s into the first spell would have been almost due; the new spell waits the full 4.5s.
      advance(4_499);
      expect(result.current.glancingUp).toBe(false);
      advance(1);
      expect(result.current.glancingUp).toBe(true);
    });

    it('does not glance while building', () => {
      const { result } = renderPose({ activity: 'building', isRunning: true });

      advance(60_000);

      expect(result.current.glancingUp).toBe(false);
    });

    it('stays on the page, without glancing, for users who prefer reduced motion', () => {
      mockReducedMotion(true);
      const { result } = renderPose({ activity: 'docs', isRunning: true });

      advance(60_000);

      expect(result.current.pose).toBe('docs');
      expect(result.current.glancingUp).toBe(false);
    });
  });
});
