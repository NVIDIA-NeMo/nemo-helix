// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act, renderHook } from '@testing-library/react';

import { useLiveSeconds } from './index';

const START = new Date('2025-01-01T00:00:00.000Z');

describe('useLiveSeconds', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('starts at the time already elapsed rather than 0', () => {
    vi.setSystemTime(new Date('2025-01-01T00:01:30.000Z'));

    const { result } = renderHook(() => useLiveSeconds({ startDate: START }));

    expect(result.current).toBe(90);
  });

  it('should update the live seconds', () => {
    vi.setSystemTime(START);
    const { result } = renderHook(() => useLiveSeconds({ startDate: START }));

    expect(result.current).toBe(0);

    // Advance timers by 1 second (triggers interval + advances system time)
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(result.current).toBe(1);

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(result.current).toBe(2);

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(result.current).toBe(3);
  });

  it('should lock the startDate value to prevent desync when startDate is provided asynchronously', () => {
    vi.setSystemTime(START);
    const { result, rerender } = renderHook(({ startDate }) => useLiveSeconds({ startDate }), {
      initialProps: { startDate: START },
    });
    expect(result.current).toBe(0);

    // Try to change the startDate - should be ignored (locked)
    rerender({ startDate: new Date('2025-01-01T00:00:05.000Z') });

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    // Still counting from the original startDate (00:00:00); now at 00:00:01
    expect(result.current).toBe(1);

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(result.current).toBe(2);
  });

  it('should handle long durations', () => {
    vi.setSystemTime(START);
    const { result } = renderHook(() => useLiveSeconds({ startDate: START }));
    for (let i = 0; i < 50; i++) {
      act(() => {
        vi.advanceTimersByTime(1000);
      });
    }
    expect(result.current).toBe(50);
  });

  it('should return undefined if startDate is not provided', () => {
    const { result } = renderHook(() => useLiveSeconds({}));
    expect(result.current).toBeUndefined();
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(result.current).toBeUndefined();
  });

  it('returns undefined while disabled, then the time already elapsed once enabled', () => {
    vi.setSystemTime(new Date('2025-01-01T00:00:10.000Z'));
    const { result, rerender } = renderHook(
      ({ enabled }) => useLiveSeconds({ startDate: START, enabled }),
      { initialProps: { enabled: false } }
    );
    expect(result.current).toBeUndefined();

    rerender({ enabled: true });

    expect(result.current).toBe(10);
  });

  it('catches up at once when re-enabled instead of resuming from the paused count', () => {
    vi.setSystemTime(START);
    const { result, rerender } = renderHook(
      ({ enabled }) => useLiveSeconds({ startDate: START, enabled }),
      { initialProps: { enabled: true } }
    );
    act(() => {
      vi.advanceTimersByTime(10_000);
    });
    expect(result.current).toBe(10);

    rerender({ enabled: false });
    act(() => {
      vi.advanceTimersByTime(60_000);
    });
    rerender({ enabled: true });

    expect(result.current).toBe(70);
  });
});
