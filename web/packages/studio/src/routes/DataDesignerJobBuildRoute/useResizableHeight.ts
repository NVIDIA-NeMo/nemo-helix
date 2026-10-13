// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type KeyboardEvent,
  type MouseEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';

const DEFAULT_STEP_PX = 24;

interface UseResizableHeightOptions {
  defaultHeight: number;
  minHeight: number;
  maxViewportRatio: number;
  step?: number;
}

interface DragStart {
  readonly pointerY: number;
  readonly height: number;
}

/** Height state for a bottom-anchored panel resized by dragging, or arrow keys, on its top edge. */
export const useResizableHeight = ({
  defaultHeight,
  minHeight,
  maxViewportRatio,
  step = DEFAULT_STEP_PX,
}: UseResizableHeightOptions) => {
  const [rawHeight, setRawHeight] = useState(defaultHeight);
  const [isDragging, setIsDragging] = useState(false);
  const [viewportHeight, setViewportHeight] = useState(() => window.innerHeight);
  const dragStartRef = useRef<DragStart | null>(null);

  useEffect(() => {
    const handleResize = () => setViewportHeight(window.innerHeight);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const getMaxHeight = useCallback(
    () => Math.max(minHeight, Math.floor(viewportHeight * maxViewportRatio)),
    [minHeight, maxViewportRatio, viewportHeight]
  );
  const clamp = useCallback(
    (value: number) => Math.max(minHeight, Math.min(getMaxHeight(), value)),
    [minHeight, getMaxHeight]
  );

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (event: globalThis.MouseEvent) => {
      const start = dragStartRef.current;
      if (start) setRawHeight(clamp(start.height + start.pointerY - event.clientY));
    };
    const handleMouseUp = () => setIsDragging(false);

    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'row-resize';
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      dragStartRef.current = null;
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, clamp]);

  const height = clamp(rawHeight);

  const onMouseDown = (event: MouseEvent<HTMLElement>) => {
    event.preventDefault();
    dragStartRef.current = { pointerY: event.clientY, height };
    setIsDragging(true);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    const delta = event.shiftKey ? step * 2 : step;
    let next: number | undefined;
    if (event.key === 'ArrowUp') next = height + delta;
    if (event.key === 'ArrowDown') next = height - delta;
    if (event.key === 'Home') next = getMaxHeight();
    if (event.key === 'End') next = minHeight;
    if (next === undefined) return;

    event.preventDefault();
    setRawHeight(clamp(next));
  };

  return { height, minHeight, maxHeight: getMaxHeight(), isDragging, onMouseDown, onKeyDown };
};
