// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  clampPosition,
  DEFAULT_FELIX_POSITION,
  getPanelPlacement,
} from '@studio/components/FelixAssistant/useFelixPosition';

const VIEWPORT = { width: 1000, height: 800 };
const FELIX = { width: 80, height: 104 };

describe('clampPosition', () => {
  it('leaves a position inside the viewport alone', () => {
    expect(clampPosition({ right: 300, bottom: 200 }, FELIX, VIEWPORT)).toEqual({
      right: 300,
      bottom: 200,
    });
  });

  it('never lets him closer than the edge margin to the bottom-right', () => {
    expect(clampPosition({ right: -50, bottom: 0 }, FELIX, VIEWPORT)).toEqual({
      right: 8,
      bottom: 8,
    });
  });

  it('keeps all of him on-screen at the top-left', () => {
    // 1000 - 80 - 8 = 912 and 800 - 104 - 8 = 688.
    expect(clampPosition({ right: 5000, bottom: 5000 }, FELIX, VIEWPORT)).toEqual({
      right: 912,
      bottom: 688,
    });
  });

  it('rounds to whole pixels', () => {
    expect(clampPosition({ right: 100.6, bottom: 50.2 }, FELIX, VIEWPORT)).toEqual({
      right: 101,
      bottom: 50,
    });
  });

  it('degrades to the edge margin when the viewport is smaller than Felix', () => {
    expect(clampPosition({ right: 300, bottom: 300 }, FELIX, { width: 40, height: 40 })).toEqual({
      right: 8,
      bottom: 8,
    });
  });
});

describe('getPanelPlacement', () => {
  it('opens above him at the default bottom-right corner, at full height', () => {
    expect(getPanelPlacement(DEFAULT_FELIX_POSITION, FELIX, VIEWPORT)).toEqual({
      side: 'top',
      maxHeight: 640,
    });
  });

  it('opens below him when he is near the top and there is more room underneath', () => {
    const placement = getPanelPlacement({ right: 16, bottom: 680 }, FELIX, VIEWPORT);

    expect(placement.side).toBe('bottom');
    // 680 (room below Felix's bottom edge) - 14 gap - 8 edge margin.
    expect(placement.maxHeight).toBe(640);
  });

  it('shrinks the panel to the room available instead of overlapping him', () => {
    // Felix mid-screen on a short viewport: neither side fits 640px, so take the bigger one.
    const placement = getPanelPlacement({ right: 16, bottom: 150 }, FELIX, {
      width: 1000,
      height: 500,
    });

    // above = 500 - 150 - 104 - 22 = 224; below = 150 - 22 = 128 -> above wins.
    expect(placement).toEqual({ side: 'top', maxHeight: 224 });
  });

  it('never makes the panel unusably short', () => {
    const placement = getPanelPlacement({ right: 16, bottom: 16 }, FELIX, {
      width: 1000,
      height: 150,
    });

    expect(placement.maxHeight).toBe(200);
  });
});
