// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ownedFirst } from '@studio/util/ownedFirst';

describe('ownedFirst', () => {
  const newestFirst = [
    { name: 'a', created_by: 'other' },
    { name: 'b', created_by: 'me' },
    { name: 'c', created_by: null },
    { name: 'd', created_by: 'me' },
  ];

  it('puts the user’s items first and keeps each group’s order', () => {
    expect(ownedFirst(newestFirst, 'me').map((i) => i.name)).toEqual(['b', 'd', 'a', 'c']);
  });

  it('leaves the order alone when signed out', () => {
    expect(ownedFirst(newestFirst, undefined).map((i) => i.name)).toEqual(['a', 'b', 'c', 'd']);
  });
});
