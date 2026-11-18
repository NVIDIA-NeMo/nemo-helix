// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type RefObject, useEffect, useState } from 'react';

const TAIL_SIZE_PX = 16;
/** Keep the tail clear of the panel's rounded corners. */
const TAIL_CORNER_INSET_PX = 16;

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

interface UseBubbleTailOptions {
  isOpen: boolean;
  anchorRef: RefObject<HTMLElement | null>;
  contentRef: RefObject<HTMLElement | null>;
}

/**
 * Measures where a speech-bubble tail should sit so it points at the anchor (Felix).
 *
 * Returns the tail's left offset within the panel, or null until the panel is open and measured.
 * The panel is shifted sideways to stay on-screen when Felix is near a left or right edge, and he
 * can be moved while it is open, so the offset is measured against the real positions each frame
 * (only while open) rather than hard-coded. State only updates when the result changes.
 */
export const useBubbleTail = ({
  isOpen,
  anchorRef,
  contentRef,
}: UseBubbleTailOptions): number | null => {
  const [left, setLeft] = useState<number | null>(null);

  useEffect(() => {
    if (!isOpen) {
      setLeft(null);
      return;
    }

    let frame = 0;
    const update = () => {
      const anchor = anchorRef.current;
      const content = contentRef.current;
      if (anchor && content) {
        const anchorRect = anchor.getBoundingClientRect();
        const contentRect = content.getBoundingClientRect();
        const next = clamp(
          anchorRect.left + anchorRect.width / 2 - contentRect.left - TAIL_SIZE_PX / 2,
          TAIL_CORNER_INSET_PX,
          Math.max(TAIL_CORNER_INSET_PX, contentRect.width - TAIL_CORNER_INSET_PX - TAIL_SIZE_PX)
        );
        setLeft((previous) =>
          previous !== null && Math.abs(previous - next) < 0.5 ? previous : next
        );
      }
      frame = requestAnimationFrame(update);
    };

    update();
    return () => cancelAnimationFrame(frame);
  }, [isOpen, anchorRef, contentRef]);

  return left;
};
