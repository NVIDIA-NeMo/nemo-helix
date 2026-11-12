// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { composePatches, mergePatch } from '@studio/providers/preferences/mergePatch';

describe('mergePatch', () => {
  it('follows RFC 7396', () => {
    expect(mergePatch({ a: 'b', c: { d: 'e', f: 'g' } }, { a: 'z', c: { f: null } })).toEqual({
      a: 'z',
      c: { d: 'e' },
    });
    expect(mergePatch({ a: [1, 2] }, { a: [3] })).toEqual({ a: [3] });
    expect(mergePatch(undefined, { a: { b: null } })).toEqual({ a: {} });
  });
});

describe('composePatches', () => {
  it('keeps the later value and preserves deletes', () => {
    expect(composePatches({ a: 1, m: { x: 1 } }, { b: null, m: { y: 2 } })).toEqual({
      a: 1,
      b: null,
      m: { x: 1, y: 2 },
    });
  });
});
