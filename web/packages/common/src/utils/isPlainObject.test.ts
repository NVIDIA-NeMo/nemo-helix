// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isPlainObject } from '@nemo/common/src/utils/isPlainObject';

describe('isPlainObject', () => {
  it('accepts keyed objects, including null-prototype objects and class instances', () => {
    expect(isPlainObject({})).toBe(true);
    expect(isPlainObject({ a: 1 })).toBe(true);
    expect(isPlainObject(Object.create(null))).toBe(true);
    expect(isPlainObject(new Date())).toBe(true);
  });

  it('rejects null, arrays and primitives', () => {
    expect(isPlainObject(null)).toBe(false);
    expect(isPlainObject([])).toBe(false);
    expect(isPlainObject([1, 2])).toBe(false);
    expect(isPlainObject('a')).toBe(false);
    expect(isPlainObject(0)).toBe(false);
    expect(isPlainObject(undefined)).toBe(false);
  });

  it('works as an array predicate', () => {
    const values: unknown[] = [{ a: 1 }, [1], null, { b: 2 }];
    expect(values.filter(isPlainObject)).toEqual([{ a: 1 }, { b: 2 }]);
    expect([{}, { a: 1 }].every(isPlainObject)).toBe(true);
  });
});
