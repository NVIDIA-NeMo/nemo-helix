// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TooltipCoords } from '@studio/components/WelcomeTour/types';
import type { TooltipPosition } from 'modern-tour';

const TOOLTIP_OFFSET = 16;
const VIEWPORT_PADDING = 16;

export function computePosition(
  targetRect: DOMRect,
  tooltipRect: DOMRect,
  position: TooltipPosition
): TooltipCoords {
  const side = position.split('-')[0];
  const alignment = position.split('-')[1] ?? '';

  let left = 0;
  let top = 0;

  switch (side) {
    case 'bottom':
      top = targetRect.bottom + TOOLTIP_OFFSET;
      left =
        alignment === 'start'
          ? targetRect.left
          : alignment === 'end'
            ? targetRect.right - tooltipRect.width
            : targetRect.left + targetRect.width / 2 - tooltipRect.width / 2;
      break;
    case 'top':
      top = targetRect.top - TOOLTIP_OFFSET - tooltipRect.height;
      left =
        alignment === 'start'
          ? targetRect.left
          : alignment === 'end'
            ? targetRect.right - tooltipRect.width
            : targetRect.left + targetRect.width / 2 - tooltipRect.width / 2;
      break;
    case 'right':
      left = targetRect.right + TOOLTIP_OFFSET;
      top =
        alignment === 'start'
          ? targetRect.top
          : alignment === 'end'
            ? targetRect.bottom - tooltipRect.height
            : targetRect.top + targetRect.height / 2 - tooltipRect.height / 2;
      break;
    case 'left':
      left = targetRect.left - TOOLTIP_OFFSET - tooltipRect.width;
      top =
        alignment === 'start'
          ? targetRect.top
          : alignment === 'end'
            ? targetRect.bottom - tooltipRect.height
            : targetRect.top + targetRect.height / 2 - tooltipRect.height / 2;
      break;
  }

  // Clamp to viewport
  left = Math.max(
    VIEWPORT_PADDING,
    Math.min(left, window.innerWidth - tooltipRect.width - VIEWPORT_PADDING)
  );
  top = Math.max(
    VIEWPORT_PADDING,
    Math.min(top, window.innerHeight - tooltipRect.height - VIEWPORT_PADDING)
  );

  return { left, top };
}
