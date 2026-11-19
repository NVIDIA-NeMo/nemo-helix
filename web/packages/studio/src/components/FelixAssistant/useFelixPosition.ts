// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useLocalStorage } from '@studio/util/hooks/useLocalStorage';
import { FELIX_POSITION_KEY } from '@studio/util/localStorage';
import {
  type KeyboardEvent,
  type PointerEvent,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';

/** Pixel offsets from the viewport's bottom-right corner, so Felix stays anchored there on resize. */
export interface FelixPosition {
  right: number;
  bottom: number;
}

interface Size {
  width: number;
  height: number;
}

export const DEFAULT_FELIX_POSITION: FelixPosition = { right: 16, bottom: 16 };

/** Felix can never be dragged closer than this to any edge of the viewport. */
const EDGE_MARGIN_PX = 8;
/** Pointer travel needed before a press counts as a drag rather than a click. */
const DRAG_THRESHOLD_PX = 5;
const KEY_STEP_PX = 16;
const KEY_STEP_LARGE_PX = 64;

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

/** localStorage is user-editable, so never trust what comes back. */
const parsePosition = (value: unknown): FelixPosition | null => {
  if (typeof value !== 'object' || value === null) return null;
  const { right, bottom } = value as Record<string, unknown>;
  return isFiniteNumber(right) && isFiniteNumber(bottom) ? { right, bottom } : null;
};

const getViewport = (): Size => ({
  // clientWidth excludes the scrollbar; innerWidth is the fallback for environments without layout.
  width: document.documentElement.clientWidth || window.innerWidth,
  height: document.documentElement.clientHeight || window.innerHeight,
});

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

/** Keeps Felix fully on-screen. Exported for tests. */
export const clampPosition = (
  position: FelixPosition,
  size: Size,
  viewport: Size
): FelixPosition => ({
  // Whole pixels keep the saved value tidy and avoid sub-pixel blur on the character.
  right: Math.round(
    clamp(
      position.right,
      EDGE_MARGIN_PX,
      Math.max(EDGE_MARGIN_PX, viewport.width - size.width - EDGE_MARGIN_PX)
    )
  ),
  bottom: Math.round(
    clamp(
      position.bottom,
      EDGE_MARGIN_PX,
      Math.max(EDGE_MARGIN_PX, viewport.height - size.height - EDGE_MARGIN_PX)
    )
  ),
});

export interface PanelPlacement {
  /** Which side of Felix the chat panel opens on. */
  side: 'top' | 'bottom';
  /** The tallest the panel can be on that side without leaving the viewport. */
  maxHeight: number;
}

const PANEL_PREFERRED_HEIGHT_PX = 640;
const PANEL_MIN_HEIGHT_PX = 200;
/** Space between Felix and the panel. Keep in sync with the panel's margin in FelixAssistant. */
export const PANEL_GAP_PX = 14;

/**
 * Chooses where the chat panel opens, from the room around Felix.
 *
 * The popover's own flipping only works when the opposite side fits the whole panel, and with a
 * tall panel neither side often does, so it would end up on top of Felix. Deciding here instead
 * lets the panel shrink to fit the side with the most room (preferring above him).
 */
export const getPanelPlacement = (
  position: FelixPosition,
  size: Size,
  viewport: Size
): PanelPlacement => {
  const roomAbove = viewport.height - position.bottom - size.height - PANEL_GAP_PX - EDGE_MARGIN_PX;
  const roomBelow = position.bottom - PANEL_GAP_PX - EDGE_MARGIN_PX;
  const side = roomAbove >= PANEL_PREFERRED_HEIGHT_PX || roomAbove >= roomBelow ? 'top' : 'bottom';
  const room = side === 'top' ? roomAbove : roomBelow;
  return {
    side,
    maxHeight: Math.round(clamp(room, PANEL_MIN_HEIGHT_PX, PANEL_PREFERRED_HEIGHT_PX)),
  };
};

interface DragState {
  pointerId: number;
  startX: number;
  startY: number;
  origin: FelixPosition;
  active: boolean;
}

interface UseFelixPositionOptions {
  /** Called once when a press turns into a drag (e.g. to close the chat panel). */
  onDragStart?: () => void;
}

/**
 * Lets the user move Felix around the screen, by dragging him or with the arrow keys.
 *
 * - The position is remembered per browser (localStorage) and clamped to the viewport, including
 *   after a window resize. A resize never rewrites the saved value, so growing the window again
 *   puts him back where the user left him.
 * - A press only becomes a drag after a small movement, so a plain click still reaches the button.
 * - Double-clicking resets him to the default corner.
 */
export const useFelixPosition = ({ onDragStart }: UseFelixPositionOptions = {}) => {
  const ref = useRef<HTMLButtonElement>(null);
  const [storedPosition, setStoredPosition, clearStoredPosition] =
    useLocalStorage<unknown>(FELIX_POSITION_KEY);
  const [dragPosition, setDragPosition] = useState<FelixPosition | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [viewport, setViewport] = useState(getViewport);
  const [size, setSize] = useState<Size>({ width: 0, height: 0 });

  const dragRef = useRef<DragState | null>(null);
  const latestDragPositionRef = useRef<FelixPosition | null>(null);
  const suppressClickRef = useRef(false);

  const position = clampPosition(
    dragPosition ?? parsePosition(storedPosition) ?? DEFAULT_FELIX_POSITION,
    size,
    viewport
  );

  useEffect(() => {
    const handleResize = () => setViewport(getViewport());
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Felix's own size is needed to keep all of him on-screen; the image loads after first paint.
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const measure = () =>
      setSize((previous) =>
        previous.width === element.offsetWidth && previous.height === element.offsetHeight
          ? previous
          : { width: element.offsetWidth, height: element.offsetHeight }
      );
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const handlePointerDown = useCallback(
    (event: PointerEvent<HTMLButtonElement>) => {
      // Primary button / touch / pen only. Capturing now means a fast flick can't outrun Felix.
      if (event.button !== 0) return;
      dragRef.current = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        origin: position,
        active: false,
      };
      try {
        event.currentTarget.setPointerCapture(event.pointerId);
      } catch {
        // Not every environment supports pointer capture; dragging still works without it.
      }
    },
    [position]
  );

  const handlePointerMove = useCallback(
    (event: PointerEvent<HTMLButtonElement>) => {
      const drag = dragRef.current;
      if (!drag || drag.pointerId !== event.pointerId) return;

      const deltaX = event.clientX - drag.startX;
      const deltaY = event.clientY - drag.startY;
      if (!drag.active) {
        if (Math.hypot(deltaX, deltaY) < DRAG_THRESHOLD_PX) return;
        drag.active = true;
        setIsDragging(true);
        onDragStart?.();
      }

      const next = clampPosition(
        { right: drag.origin.right - deltaX, bottom: drag.origin.bottom - deltaY },
        size,
        viewport
      );
      latestDragPositionRef.current = next;
      setDragPosition(next);
    },
    [onDragStart, size, viewport]
  );

  const finishDrag = useCallback(
    (event: PointerEvent<HTMLButtonElement>, commit: boolean) => {
      const drag = dragRef.current;
      if (!drag || drag.pointerId !== event.pointerId) return;
      dragRef.current = null;
      try {
        event.currentTarget.releasePointerCapture(event.pointerId);
      } catch {
        // Capture may already be released.
      }
      if (!drag.active) return;

      if (commit && latestDragPositionRef.current) setStoredPosition(latestDragPositionRef.current);
      latestDragPositionRef.current = null;
      setDragPosition(null);
      setIsDragging(false);

      // The browser fires a click right after the pointer is released; it must not toggle the chat.
      suppressClickRef.current = true;
      setTimeout(() => {
        suppressClickRef.current = false;
      }, 0);
    },
    [setStoredPosition]
  );

  const handlePointerUp = useCallback(
    (event: PointerEvent<HTMLButtonElement>) => finishDrag(event, true),
    [finishDrag]
  );
  const handlePointerCancel = useCallback(
    (event: PointerEvent<HTMLButtonElement>) => finishDrag(event, false),
    [finishDrag]
  );

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLButtonElement>) => {
      const step = event.shiftKey ? KEY_STEP_LARGE_PX : KEY_STEP_PX;
      // Moving Felix left/up grows his offset from the right/bottom edge.
      const deltas: Record<string, FelixPosition> = {
        ArrowLeft: { right: step, bottom: 0 },
        ArrowRight: { right: -step, bottom: 0 },
        ArrowUp: { right: 0, bottom: step },
        ArrowDown: { right: 0, bottom: -step },
      };
      const delta = deltas[event.key];
      if (!delta) return;

      event.preventDefault();
      setStoredPosition(
        clampPosition(
          { right: position.right + delta.right, bottom: position.bottom + delta.bottom },
          size,
          viewport
        )
      );
    },
    [position, setStoredPosition, size, viewport]
  );

  /** True (once) if the click that follows a drag should be ignored. */
  const consumeDragClick = useCallback(() => {
    if (!suppressClickRef.current) return false;
    suppressClickRef.current = false;
    return true;
  }, []);

  return {
    ref,
    position,
    placement: getPanelPlacement(position, size, viewport),
    isDragging,
    consumeDragClick,
    handlers: {
      onPointerDown: handlePointerDown,
      onPointerMove: handlePointerMove,
      onPointerUp: handlePointerUp,
      onPointerCancel: handlePointerCancel,
      onKeyDown: handleKeyDown,
      onDoubleClick: clearStoredPosition,
    },
  };
};
