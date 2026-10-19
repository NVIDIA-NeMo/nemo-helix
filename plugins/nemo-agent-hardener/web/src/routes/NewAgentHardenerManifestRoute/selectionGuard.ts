// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Guards async agent/project selection callbacks against a stale one landing after a newer
 * selection — or after the user cleared the selection outright — and overwriting current form
 * state with outdated data.
 */
export class SelectionGuard {
  private generation = 0;

  /** Start a new selection; use the returned id with `isCurrent` in that selection's callbacks. */
  begin(): number {
    this.generation += 1;
    return this.generation;
  }

  /** True if `generation` is still the most recent selection — false once superseded or cleared. */
  isCurrent(generation: number): boolean {
    return generation === this.generation;
  }

  /** Invalidate the current selection without starting a new one, e.g. the user removed the file. */
  invalidate(): void {
    this.generation += 1;
  }
}
