// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SelectionGuard } from '@agent-hardener/routes/NewAgentHardenerManifestRoute/selectionGuard';

describe('SelectionGuard', () => {
  it('is current for the selection that just began, with nothing superseding it', () => {
    const guard = new SelectionGuard();
    const generation = guard.begin();

    expect(guard.isCurrent(generation)).toBe(true);
  });

  it('marks an earlier selection stale once a newer one begins', () => {
    const guard = new SelectionGuard();
    const first = guard.begin();
    const second = guard.begin();

    expect(guard.isCurrent(first)).toBe(false);
    expect(guard.isCurrent(second)).toBe(true);
  });

  it('marks the in-flight selection stale when the user removes it mid-upload', () => {
    // Mirrors onRemoveFile: the upload/inspect callbacks already captured `generation` and must
    // not act on it once the selection is invalidated, even though no *new* selection began.
    const guard = new SelectionGuard();
    const generation = guard.begin();

    guard.invalidate();

    expect(guard.isCurrent(generation)).toBe(false);
  });

  it('accepts a selection that begins again after an invalidation', () => {
    const guard = new SelectionGuard();
    guard.begin();
    guard.invalidate();
    const generation = guard.begin();

    expect(guard.isCurrent(generation)).toBe(true);
  });
});
