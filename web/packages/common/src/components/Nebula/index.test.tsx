// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Nebula } from '@nemo/common/src/components/Nebula';
import { render } from '@testing-library/react';

const makeCanvasContext = (): CanvasRenderingContext2D =>
  ({
    arc: vi.fn(),
    beginPath: vi.fn(),
    clearRect: vi.fn(),
    closePath: vi.fn(),
    fill: vi.fn(),
    lineTo: vi.fn(),
    moveTo: vi.fn(),
    restore: vi.fn(),
    rotate: vi.fn(),
    save: vi.fn(),
    stroke: vi.fn(),
    translate: vi.fn(),
  }) as unknown as CanvasRenderingContext2D;

describe('Nebula', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('does not retain animation frames across remounts', () => {
    const pendingFrames = new Set<number>();
    let nextFrameId = 0;

    vi.spyOn(window, 'requestAnimationFrame').mockImplementation(() => {
      nextFrameId += 1;
      pendingFrames.add(nextFrameId);
      return nextFrameId;
    });
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation((frameId) => {
      pendingFrames.delete(frameId);
    });
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(() =>
      makeCanvasContext()
    );

    for (let cycle = 0; cycle < 3; cycle += 1) {
      const { unmount } = render(<Nebula variant="sphere" />);

      unmount();
    }

    expect(pendingFrames.size).toBe(0);
    expect(window.cancelAnimationFrame).toHaveBeenCalledTimes(3);
  });
});
