// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import cn from 'classnames';
import { GripVertical } from 'lucide-react';
import {
  type FC,
  type KeyboardEvent,
  type MouseEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';

const DEFAULT_RESIZE_STEP_PX = 24;
const DIVIDER_WIDTH_PX = 12;

interface PanelBounds {
  readonly containerLeft: number;
  readonly maxLeftWidth: number;
}

interface ResizeablePanelProps {
  slotLeft: ReactNode;
  slotRight: ReactNode;
  defaultLeftWidth?: number;
  minLeftWidth?: number;
  minRightWidth?: number;
  maxLeftWidth?: number;
  resizeStep?: number;
  separatorLabel?: string;
  variant?: 'panel' | 'plain';
  leftClassName?: string;
  rightClassName?: string;
  separatorClassName?: string;
  className?: string;
}

export const ResizeablePanel: FC<ResizeablePanelProps> = ({
  slotLeft,
  slotRight,
  defaultLeftWidth = 410,
  minLeftWidth = 200,
  minRightWidth = minLeftWidth,
  maxLeftWidth,
  resizeStep = DEFAULT_RESIZE_STEP_PX,
  separatorLabel = 'Resize panels',
  variant = 'panel',
  leftClassName,
  rightClassName,
  separatorClassName,
  className,
}) => {
  const [leftWidth, setLeftWidth] = useState(Math.max(minLeftWidth, defaultLeftWidth));
  const [resolvedMaxLeftWidth, setResolvedMaxLeftWidth] = useState(
    maxLeftWidth ?? Number.MAX_SAFE_INTEGER
  );
  const [isDragging, setIsDragging] = useState(false);
  const [isStacked, setIsStacked] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const separatorRef = useRef<HTMLDivElement>(null);
  const dragBoundsRef = useRef<PanelBounds | null>(null);

  const measurePanelBounds = useCallback((): PanelBounds => {
    const containerRect = containerRef.current?.getBoundingClientRect();
    const containerWidth = containerRect?.width ?? 0;
    if (containerWidth <= 0) {
      return {
        containerLeft: containerRect?.left ?? 0,
        maxLeftWidth: Math.max(minLeftWidth, maxLeftWidth ?? Number.MAX_SAFE_INTEGER),
      };
    }

    const separatorWidth = separatorRef.current?.offsetWidth ?? DIVIDER_WIDTH_PX;
    const availableWidth = containerWidth - minRightWidth - separatorWidth;
    return {
      containerLeft: containerRect?.left ?? 0,
      maxLeftWidth: Math.max(
        minLeftWidth,
        maxLeftWidth === undefined ? availableWidth : Math.min(maxLeftWidth, availableWidth)
      ),
    };
  }, [maxLeftWidth, minLeftWidth, minRightWidth]);

  useEffect(() => {
    const updatePanelBounds = () => {
      const { maxLeftWidth: nextMaxWidth } = measurePanelBounds();
      setResolvedMaxLeftWidth(nextMaxWidth);
      setLeftWidth((width) => Math.min(width, nextMaxWidth));
    };

    updatePanelBounds();

    if (!containerRef.current || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(updatePanelBounds);
    observer.observe(containerRef.current);
    if (separatorRef.current) observer.observe(separatorRef.current);
    return () => observer.disconnect();
  }, [measurePanelBounds]);

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (event: globalThis.MouseEvent) => {
      const { containerLeft, maxLeftWidth: nextMaxWidth } = dragBoundsRef.current ?? {
        containerLeft: 0,
        maxLeftWidth: minLeftWidth,
      };
      setLeftWidth(Math.max(minLeftWidth, Math.min(nextMaxWidth, event.clientX - containerLeft)));
    };
    const handleMouseUp = () => {
      dragBoundsRef.current = null;
      setIsDragging(false);
    };

    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'col-resize';
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      dragBoundsRef.current = null;
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, minLeftWidth]);

  const handleMouseDown = (event: MouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    const bounds = measurePanelBounds();
    dragBoundsRef.current = bounds;
    setResolvedMaxLeftWidth(bounds.maxLeftWidth);
    setIsDragging(true);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? resizeStep * 2 : resizeStep;
    let nextWidth: number | undefined;
    if (event.key === 'ArrowLeft') nextWidth = leftWidth - step;
    if (event.key === 'ArrowRight') nextWidth = leftWidth + step;
    if (event.key === 'Home') nextWidth = minLeftWidth;
    if (event.key === 'End') nextWidth = resolvedMaxLeftWidth;
    if (nextWidth === undefined) return;

    event.preventDefault();
    setLeftWidth(Math.max(minLeftWidth, Math.min(resolvedMaxLeftWidth, nextWidth)));
  };

  useEffect(() => {
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(([entry]) => {
      const separatorWidth = separatorRef.current?.offsetWidth ?? DIVIDER_WIDTH_PX;
      const stacked = entry.contentRect.width < minLeftWidth + minRightWidth + separatorWidth;
      setIsStacked(stacked);
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [minLeftWidth, minRightWidth]);

  const plain = variant === 'plain';

  return (
    <div
      ref={containerRef}
      className={cn('flex w-full', isStacked ? 'flex-col' : 'h-full', className)}
    >
      <div
        // eslint-disable-next-line no-restricted-syntax
        style={{ width: isStacked ? '100%' : leftWidth }}
        className={cn(
          'shrink-0',
          !plain &&
            'overflow-y-auto rounded-bl-xl rounded-tl-xl border border-base bg-surface-raised',
          leftClassName
        )}
      >
        {slotLeft}
      </div>

      {!isStacked ? (
        <div
          ref={separatorRef}
          role="separator"
          aria-orientation="vertical"
          aria-label={separatorLabel}
          aria-valuemin={minLeftWidth}
          aria-valuemax={Math.round(resolvedMaxLeftWidth)}
          aria-valuenow={Math.round(leftWidth)}
          tabIndex={0}
          className={cn(
            'group relative shrink-0 cursor-col-resize items-center justify-center focus-visible:outline-none',
            plain
              ? 'flex w-[var(--spacing-density-md)]'
              : 'flex w-3 border-y border-base bg-surface-raised',
            !plain && isDragging && 'bg-surface-hover',
            separatorClassName
          )}
          onMouseDown={handleMouseDown}
          onKeyDown={handleKeyDown}
        >
          {plain ? (
            <span
              className={cn(
                'flex h-10 w-1 items-center justify-center rounded-full bg-border-base transition-colors group-hover:bg-border-strong group-focus-visible:bg-border-brand',
                isDragging && 'bg-border-brand'
              )}
            >
              <GripVertical aria-hidden className="size-3 max-w-none text-secondary" />
            </span>
          ) : (
            <>
              <div
                className={cn(
                  'absolute inset-y-0 left-[5px] w-px bg-border-base transition-colors',
                  'group-hover:bg-border-strong',
                  isDragging && 'bg-border-brand'
                )}
              />
              <GripVertical
                className={cn(
                  'relative z-10 size-3 text-content-secondary transition-opacity',
                  'opacity-0 group-hover:opacity-100',
                  isDragging && 'opacity-100 text-content-brand'
                )}
                aria-hidden
              />
            </>
          )}
        </div>
      ) : null}

      <div
        className={cn(
          'min-w-0 flex-1',
          !plain &&
            'overflow-hidden rounded-br-xl rounded-tr-xl border border-base bg-surface-raised',
          rightClassName
        )}
      >
        {slotRight}
      </div>
    </div>
  );
};
