// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HourglassMid } from '@nemo/icons/icons';
import { Flex, Text } from '@nvidia/foundations-react-core';
import { useVirtualizer } from '@tanstack/react-virtual';
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';

const LIST_MAX_HEIGHT = 300;
const OVERSCAN = 8;
const LOAD_MORE_THRESHOLD = 5;

/** Assumed viewport before the first measurement, so the initial render is already windowed. */
const INITIAL_RECT = { width: 360, height: LIST_MAX_HEIGHT };

export interface VirtualDropdownListProps<R extends { key: string }> {
  /** Headings and items share one flat index space. */
  rows: readonly R[];
  renderRow: (row: R) => ReactNode;
  /** First-guess row height in px; rows measure themselves once mounted. */
  estimateSize: (row: R) => number;
  loading?: boolean;
  /** Called as the user scrolls near the end; no-op when {@link hasMore} is false. */
  onLoadMore?: () => void | Promise<void>;
  hasMore?: boolean;
  isLoadingMore?: boolean;
  /** Footer copy once every page has loaded. Omit to render nothing. */
  doneLoadingMessage?: string;
  emptyMessage: string;
  loadingMessage: string;
}

/**
 * The scrollable body of a searchable dropdown, virtualized so only the visible slice of a long,
 * paged catalogue is in the DOM, and asking for the next page as the user nears the end.
 *
 * Rows that carry hover submenus mount a details panel each, so an unvirtualized list of a few
 * hundred entries mounts thousands of nodes on open — this keeps that cost proportional to the
 * viewport instead of the catalogue.
 */
export function VirtualDropdownList<R extends { key: string }>({
  rows,
  renderRow,
  estimateSize,
  loading = false,
  onLoadMore,
  hasMore = false,
  isLoadingMore = false,
  doneLoadingMessage,
  emptyMessage,
  loadingMessage,
}: VirtualDropdownListProps<R>) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [viewportHeight, setViewportHeight] = useState(0);
  const [isLoadingMoreLocal, setIsLoadingMoreLocal] = useState(false);

  useLayoutEffect(() => {
    const element = scrollRef.current;
    if (!element) return;

    const measure = () => setViewportHeight(element.clientHeight);
    measure();

    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const isVirtualized = viewportHeight > 0;

  // TanStack rebuilds every row's measurements when `getItemKey` changes identity, so keep it stable.
  const getItemKey = useCallback((index: number) => rows[index].key, [rows]);

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: (index) => estimateSize(rows[index]),
    getItemKey,
    overscan: OVERSCAN,
    initialRect: INITIAL_RECT,
  });

  const virtualItems = virtualizer.getVirtualItems();
  const lastVisibleIndex =
    virtualItems.length > 0 ? virtualItems[virtualItems.length - 1].index : -1;

  // Refs keep `loadMore` stable so a failed page doesn't re-fire the effect as loading state flips.
  const latest = useRef({ onLoadMore, hasMore, isLoadingMore });
  const inFlight = useRef(false);
  useLayoutEffect(() => {
    latest.current = { onLoadMore, hasMore, isLoadingMore };
  }, [onLoadMore, hasMore, isLoadingMore]);

  const loadMore = useCallback(async () => {
    const { onLoadMore: load, hasMore: more, isLoadingMore: busy } = latest.current;
    if (!load || !more || busy || inFlight.current) return;
    inFlight.current = true;
    setIsLoadingMoreLocal(true);
    try {
      await load();
    } finally {
      inFlight.current = false;
      setIsLoadingMoreLocal(false);
    }
  }, []);

  useEffect(() => {
    if (lastVisibleIndex < 0) return;
    if (lastVisibleIndex >= rows.length - LOAD_MORE_THRESHOLD) {
      loadMore().catch(() => undefined);
    }
  }, [hasMore, lastVisibleIndex, loadMore, rows.length]);

  const loadingMore = isLoadingMore || isLoadingMoreLocal;
  const showDoneMessage = Boolean(doneLoadingMessage) && !hasMore && !loading && !loadingMore;

  // The scroll container always renders so `scrollRef` is set on mount, even when the list starts empty.
  if (rows.length === 0) {
    return (
      <div ref={scrollRef} className="w-full overflow-auto max-h-[300px]">
        <Flex align="center" justify="center" className="w-full px-density-md py-density-lg">
          <Text className="text-secondary">
            {loading || loadingMore ? loadingMessage : emptyMessage}
          </Text>
        </Flex>
      </div>
    );
  }

  return (
    <div ref={scrollRef} className="w-full overflow-auto max-h-[300px]">
      {isVirtualized ? (
        <div
          className="relative w-full"
          // eslint-disable-next-line no-restricted-syntax -- runtime pixel height from the virtualizer
          style={{ height: virtualizer.getTotalSize() }}
        >
          {virtualItems.map((virtualRow) => (
            <div
              key={virtualRow.key}
              ref={virtualizer.measureElement}
              data-index={virtualRow.index}
              className="absolute top-0 left-0 w-full"
              // eslint-disable-next-line no-restricted-syntax -- per-scroll offset from the virtualizer
              style={{ transform: `translateY(${virtualRow.start}px)` }}
            >
              {renderRow(rows[virtualRow.index])}
            </div>
          ))}
        </div>
      ) : (
        rows.map((row) => <div key={row.key}>{renderRow(row)}</div>)
      )}

      {loadingMore && (
        <Flex align="center" justify="center" gap="density-sm" className="w-full py-density-sm">
          <HourglassMid size="14" className="animate-spin" aria-hidden />
          <Text className="text-secondary" kind="body/regular/sm">
            Loading more…
          </Text>
        </Flex>
      )}
      {showDoneMessage && (
        <Flex align="center" justify="center" className="w-full py-density-sm">
          <Text className="text-secondary" kind="body/regular/sm">
            {doneLoadingMessage}
          </Text>
        </Flex>
      )}
    </div>
  );
}
