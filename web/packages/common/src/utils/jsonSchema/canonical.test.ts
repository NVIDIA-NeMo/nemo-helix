// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { canonicalJson } from '@nemo/common/src/utils/jsonSchema/canonical';

describe('canonicalJson', () => {
  it('serializes primitives like JSON.stringify', () => {
    expect(canonicalJson(null)).toBe('null');
    expect(canonicalJson('abc')).toBe('"abc"');
    expect(canonicalJson(42)).toBe('42');
    expect(canonicalJson(true)).toBe('true');
  });

  it('produces the same string for objects whose keys are in different orders', () => {
    expect(canonicalJson({ b: 1, a: 2 })).toBe(canonicalJson({ a: 2, b: 1 }));
  });

  it('serializes arrays preserving element order', () => {
    expect(canonicalJson([1, 2, 3])).toBe('[1,2,3]');
    expect(canonicalJson([1, 2, 3])).not.toBe(canonicalJson([3, 2, 1]));
  });

  it('recursively sorts nested object keys', () => {
    const a = { outer: { z: 1, a: 2 }, alpha: [{ y: 1, x: 2 }] };
    const b = { alpha: [{ x: 2, y: 1 }], outer: { a: 2, z: 1 } };
    expect(canonicalJson(a)).toBe(canonicalJson(b));
  });

  it('distinguishes structurally different values', () => {
    expect(canonicalJson({ a: 1 })).not.toBe(canonicalJson({ a: 2 }));
    expect(canonicalJson({ a: 1 })).not.toBe(canonicalJson({ a: 1, b: 2 }));
  });
});
