// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ownedFirst } from '@studio/util/ownedFirst';

const item = (id: string) => ({ id });

describe('ownedFirst', () => {
  it('puts owned items first, without repeating them, and keeps each list’s order', () => {
    const newestFirst = ['a', 'b', 'c', 'd'].map(item);
    const owned = ['d', 'b'].map(item); // also newest first
    expect(ownedFirst(owned, newestFirst).map((i) => i.id)).toEqual(['d', 'b', 'a', 'c']);
  });

  it('keeps an owned item the page of all items missed (an older dataset beyond the first page)', () => {
    const firstPage = ['a', 'b'].map(item);
    const owned = [item('old-mine')];
    expect(ownedFirst(owned, firstPage).map((i) => i.id)).toEqual(['old-mine', 'a', 'b']);
  });

  it('returns the list as is when the user owns nothing', () => {
    expect(ownedFirst([], ['a', 'b'].map(item)).map((i) => i.id)).toEqual(['a', 'b']);
  });
});
