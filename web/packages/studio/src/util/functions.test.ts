// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isDefined } from '@nemo/common/src/utils/isDefined';
import {
  assertUnreachable,
  debounceAsyncRequest,
  isNaNScore,
  isPowerOf,
  isScalar,
} from '@studio/util/functions';

describe('isDefined', () => {
  it('should return true if the value is defined', () => {
    expect(isDefined(1)).toBe(true);
    expect(isDefined('hello')).toBe(true);
    expect(isDefined(true)).toBe(true);
    expect(isDefined({})).toBe(true);
  });
  it('should return false if the value is not defined', () => {
    expect(isDefined(undefined)).toBe(false);
    expect(isDefined(null)).toBe(false);
  });
});

describe('isPowerOf', () => {
  it('should return true if the number is a power of two', () => {
    expect(isPowerOf(1)).toBe(true);
    expect(isPowerOf(2)).toBe(true);
    expect(isPowerOf(4, 2)).toBe(true);
    expect(isPowerOf(8, 2)).toBe(true);
    expect(isPowerOf(16, 2)).toBe(true);
    expect(isPowerOf(32)).toBe(true);
    expect(isPowerOf(64)).toBe(true);
  });
  it('should return false if the number is not a power of two', () => {
    expect(isPowerOf(0)).toBe(false);
    expect(isPowerOf(5)).toBe(false);
    expect(isPowerOf(6)).toBe(false);
    expect(isPowerOf(7)).toBe(false);
    expect(isPowerOf(9)).toBe(false);
  });
});

describe('debounceAsyncRequest', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should call the function after the delay', () => {
    const fn = vi.fn().mockResolvedValue(undefined);
    const debounced = debounceAsyncRequest(fn, 500);
    debounced('arg1');
    expect(fn).not.toHaveBeenCalled();
    vi.advanceTimersByTime(500);
    expect(fn).toHaveBeenCalledWith('arg1');
  });

  it('should debounce multiple calls and only invoke the last one', () => {
    const fn = vi.fn().mockResolvedValue(undefined);
    const debounced = debounceAsyncRequest(fn, 500);
    debounced('first');
    debounced('second');
    debounced('third');
    vi.advanceTimersByTime(500);
    expect(fn).toHaveBeenCalledTimes(1);
    expect(fn).toHaveBeenCalledWith('third');
  });

  it('should use default delay of 2000ms', () => {
    const fn = vi.fn().mockResolvedValue(undefined);
    const debounced = debounceAsyncRequest(fn);
    debounced();
    vi.advanceTimersByTime(1999);
    expect(fn).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(fn).toHaveBeenCalledTimes(1);
  });
});

describe('assertUnreachable', () => {
  it('should throw with default message', () => {
    expect(() => assertUnreachable('unexpected' as never)).toThrow(
      'Unknown state: "unexpected". This should never happen.'
    );
  });

  it('should throw with custom message', () => {
    expect(() => assertUnreachable('unexpected' as never, 'Custom error')).toThrow('Custom error');
  });
});

describe('isScalar', () => {
  it('accepts strings, numbers and booleans', () => {
    expect(isScalar('a')).toBe(true);
    expect(isScalar(0)).toBe(true);
    expect(isScalar(false)).toBe(true);
  });

  it('rejects null, undefined, objects and arrays', () => {
    expect(isScalar(null)).toBe(false);
    expect(isScalar(undefined)).toBe(false);
    expect(isScalar({})).toBe(false);
    expect(isScalar([])).toBe(false);
  });
});

describe('isNaNScore', () => {
  it("detects the evaluator's string NaN in any casing", () => {
    expect(isNaNScore('NaN')).toBe(true);
    expect(isNaNScore('nan')).toBe(true);
  });

  it('detects non-finite numbers', () => {
    expect(isNaNScore(Number.NaN)).toBe(true);
    expect(isNaNScore(Infinity)).toBe(true);
  });

  it('leaves real scores and other strings alone', () => {
    expect(isNaNScore(0)).toBe(false);
    expect(isNaNScore(1)).toBe(false);
    expect(isNaNScore('phishing')).toBe(false);
    expect(isNaNScore(null)).toBe(false);
  });
});
