// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isDefined } from '@nemo/common/src/utils/isDefined';

describe('isDefined', () => {
  it('should return false for null values', () => {
    expect(isDefined(null)).toBe(false);
  });

  it('should return false for undefined values', () => {
    expect(isDefined(undefined)).toBe(false);
  });

  it('should return true for string values', () => {
    expect(isDefined('hello')).toBe(true);
    expect(isDefined('')).toBe(true); // empty string is still defined
  });

  it('should return true for number values', () => {
    expect(isDefined(0)).toBe(true);
    expect(isDefined(42)).toBe(true);
    expect(isDefined(-1)).toBe(true);
    expect(isDefined(NaN)).toBe(true); // NaN is still defined
    expect(isDefined(Infinity)).toBe(true);
  });

  it('should return true for boolean values', () => {
    expect(isDefined(true)).toBe(true);
    expect(isDefined(false)).toBe(true);
  });

  it('should return true for object values', () => {
    expect(isDefined({})).toBe(true);
    expect(isDefined({ key: 'value' })).toBe(true);
    expect(isDefined([])).toBe(true);
    expect(isDefined([1, 2, 3])).toBe(true);
  });

  it('should return true for function values', () => {
    expect(isDefined(() => {})).toBe(true);
    expect(isDefined(function () {})).toBe(true);
  });
});
