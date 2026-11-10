// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useDeferredUnmount } from '@nemo/common/src/hooks/useDeferredUnmount/index';
import { renderHook, act } from '@testing-library/react';

describe('useDeferredUnmount', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('initial state', () => {
    it('isOpen is false initially', () => {
      const { result } = renderHook(() => useDeferredUnmount<string>());
      expect(result.current.isOpen).toBe(false);
    });

    it('value is null initially', () => {
      const { result } = renderHook(() => useDeferredUnmount<string>());
      expect(result.current.value).toBeNull();
    });
  });

  describe('opening (open())', () => {
    it('open(value) sets isOpen to true immediately', () => {
      const { result } = renderHook(() => useDeferredUnmount<string>());

      act(() => {
        result.current.open('test-id');
      });

      expect(result.current.isOpen).toBe(true);
    });

    it('open(value) sets value to the provided value immediately', () => {
      const { result } = renderHook(() => useDeferredUnmount<string>());

      act(() => {
        result.current.open('test-id');
      });

      expect(result.current.value).toBe('test-id');
    });

    it('open(newValue) while already open updates value without closing', () => {
      const { result } = renderHook(() => useDeferredUnmount<string>());

      act(() => {
        result.current.open('first-id');
      });

      act(() => {
        result.current.open('second-id');
      });

      expect(result.current.value).toBe('second-id');
      expect(result.current.isOpen).toBe(true);
    });
  });

  describe('closing (close())', () => {
    it('close() sets isOpen to false immediately', () => {
      const { result } = renderHook(() => useDeferredUnmount<string>());

      act(() => {
        result.current.open('test-id');
      });

      act(() => {
        result.current.close();
      });

      expect(result.current.isOpen).toBe(false);
    });

    it('close() keeps value unchanged immediately (not null)', () => {
      const { result } = renderHook(() => useDeferredUnmount<string>());

      act(() => {
        result.current.open('test-id');
      });

      act(() => {
        result.current.close();
      });

      expect(result.current.value).toBe('test-id');
    });

    it('close() sets value to null after the configured delay', () => {
      const { result } = renderHook(() => useDeferredUnmount<string>());

      act(() => {
        result.current.open('test-id');
      });

      act(() => {
        result.current.close();
      });

      expect(result.current.value).toBe('test-id');

      act(() => {
        vi.advanceTimersByTime(300);
      });

      expect(result.current.value).toBeNull();
    });

    it('delay timing is respected (custom delay)', () => {
      const { result } = renderHook(() => useDeferredUnmount<string>({ delay: 500 }));

      act(() => {
        result.current.open('test-id');
      });

      act(() => {
        result.current.close();
      });

      act(() => {
        vi.advanceTimersByTime(300);
      });

      expect(result.current.value).toBe('test-id');

      act(() => {
        vi.advanceTimersByTime(200);
      });

      expect(result.current.value).toBeNull();
    });
  });

  describe('cancellation (re-opening during close animation)', () => {
    it('open(value) during pending close cancels the delayed value clear', () => {
      const { result } = renderHook(() => useDeferredUnmount<string>());

      act(() => {
        result.current.open('first-id');
      });

      act(() => {
        result.current.close();
      });

      act(() => {
        result.current.open('second-id');
      });

      // Advance past the original timeout
      act(() => {
        vi.advanceTimersByTime(500);
      });

      // Value should NOT be cleared
      expect(result.current.value).toBe('second-id');
      expect(result.current.isOpen).toBe(true);
    });

    it('open(value) during pending close sets new value immediately', () => {
      const { result } = renderHook(() => useDeferredUnmount<string>());

      act(() => {
        result.current.open('first-id');
      });

      act(() => {
        result.current.close();
      });

      act(() => {
        result.current.open('second-id');
      });

      expect(result.current.value).toBe('second-id');
    });
  });

  describe('onOpenChange handler', () => {
    it('onOpenChange(false) behaves same as close()', () => {
      const { result } = renderHook(() => useDeferredUnmount<string>());

      act(() => {
        result.current.open('test-id');
      });

      act(() => {
        result.current.onOpenChange(false);
      });

      expect(result.current.isOpen).toBe(false);
      expect(result.current.value).toBe('test-id');

      act(() => {
        vi.advanceTimersByTime(300);
      });

      expect(result.current.value).toBeNull();
    });

    it('onOpenChange(true) sets isOpen to true (without changing value)', () => {
      const { result } = renderHook(() => useDeferredUnmount<string>());

      act(() => {
        result.current.onOpenChange(true);
      });

      expect(result.current.isOpen).toBe(true);
      expect(result.current.value).toBeNull();
    });

    it('onOpenChange(true) cancels any pending close timeout', () => {
      const { result } = renderHook(() => useDeferredUnmount<string>());

      act(() => {
        result.current.open('test-id');
      });

      act(() => {
        result.current.close();
      });

      act(() => {
        result.current.onOpenChange(true);
      });

      act(() => {
        vi.advanceTimersByTime(500);
      });

      expect(result.current.value).toBe('test-id');
    });
  });

  describe('cleanup', () => {
    it('unmounting during pending close clears the timeout (no memory leak)', () => {
      const clearTimeoutSpy = vi.spyOn(global, 'clearTimeout');
      const { result, unmount } = renderHook(() => useDeferredUnmount<string>());

      act(() => {
        result.current.open('test-id');
      });

      act(() => {
        result.current.close();
      });

      unmount();

      expect(clearTimeoutSpy).toHaveBeenCalled();
      clearTimeoutSpy.mockRestore();
    });
  });

  describe('edge cases', () => {
    it('multiple rapid open() → close() → open() cycles work correctly', () => {
      const { result } = renderHook(() => useDeferredUnmount<string>());

      act(() => {
        result.current.open('id-1');
        result.current.close();
        result.current.open('id-2');
        result.current.close();
        result.current.open('id-3');
      });

      expect(result.current.isOpen).toBe(true);
      expect(result.current.value).toBe('id-3');

      act(() => {
        vi.advanceTimersByTime(500);
      });

      // Should still have the value since we ended with open
      expect(result.current.value).toBe('id-3');
    });

    it('open() with same value as current value works correctly', () => {
      const { result } = renderHook(() => useDeferredUnmount<string>());

      act(() => {
        result.current.open('same-id');
      });

      act(() => {
        result.current.open('same-id');
      });

      expect(result.current.isOpen).toBe(true);
      expect(result.current.value).toBe('same-id');
    });

    it('works with different generic types (object)', () => {
      const { result } = renderHook(() => useDeferredUnmount<{ id: number; name: string }>());

      const testObj = { id: 1, name: 'test' };

      act(() => {
        result.current.open(testObj);
      });

      expect(result.current.value).toEqual(testObj);
    });
  });

  describe('delay configuration', () => {
    it('default delay of 300ms is used when no options provided', () => {
      const { result } = renderHook(() => useDeferredUnmount<string>());

      act(() => {
        result.current.open('test-id');
      });

      act(() => {
        result.current.close();
      });

      act(() => {
        vi.advanceTimersByTime(299);
      });

      expect(result.current.value).toBe('test-id');

      act(() => {
        vi.advanceTimersByTime(1);
      });

      expect(result.current.value).toBeNull();
    });

    it('delay: 0 clears value immediately after close', () => {
      const { result } = renderHook(() => useDeferredUnmount<string>({ delay: 0 }));

      act(() => {
        result.current.open('test-id');
      });

      act(() => {
        result.current.close();
      });

      act(() => {
        vi.advanceTimersByTime(0);
      });

      expect(result.current.value).toBeNull();
    });
  });
});
