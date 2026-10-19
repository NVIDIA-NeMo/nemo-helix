// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelWorkspaceGroup } from '@nemo/common/src/api/models/useModels';
import { ModelDropdownItem } from '@nemo/common/src/components/ModelSelectV2/ModelDropdownItem';
import type { ModelSelection } from '@nemo/common/src/components/ModelSelectV2/types';
import { creatorToIcon } from '@nemo/common/src/constants/modelMetadata';
import { getURNFromNamedEntityRef } from '@nemo/common/src/namedEntity';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { DropdownHeading, Flex, Text } from '@nvidia/foundations-react-core';
import { useVirtualizer } from '@tanstack/react-virtual';
import { LoaderCircle } from 'lucide-react';
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type FC } from 'react';

/** Rows the virtualizer measures: group headings and models share one flat index space. */
type ModelRow =
  | { kind: 'heading'; key: string; workspace: string }
  | { kind: 'model'; key: string; model: ModelEntity; urn: string };

const HEADING_HEIGHT = 32;
const ITEM_HEIGHT = 36;
const LIST_MAX_HEIGHT = 300;
const OVERSCAN = 8;
const LOAD_MORE_THRESHOLD = 5;

/** Assumed viewport before the first measurement, so the initial render is already windowed. */
const INITIAL_RECT = { width: 360, height: LIST_MAX_HEIGHT };

export interface ModelDropdownListProps {
  groups: ModelWorkspaceGroup[];
  value: ModelSelection | null;
  onSelect: (selection: ModelSelection) => void;
  hideAdapters?: boolean;
  loading?: boolean;
  /** Called as the user scrolls near the end; no-op when {@link hasMore} is false. */
  onLoadMore?: () => void | Promise<void>;
  hasMore?: boolean;
  isLoadingMore?: boolean;
  /** Footer copy once every page has loaded. Omit to render nothing. */
  doneLoadingMessage?: string;
  emptyMessage?: string;
}

/**
 * The scrollable body of the model dropdown, virtualized so only the visible slice of a
 * workspace's models is in the DOM. Each item carries a submenu with a details panel, so an
 * unvirtualized list of a few hundred models mounts thousands of nodes on open — this keeps
 * that cost proportional to the viewport instead of the catalogue.
 */
export const ModelDropdownList: FC<ModelDropdownListProps> = ({
  groups,
  value,
  onSelect,
  hideAdapters = false,
  loading = false,
  onLoadMore,
  hasMore = false,
  isLoadingMore = false,
  doneLoadingMessage,
  emptyMessage = 'No models found',
}) => {
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

  const rows = useMemo<ModelRow[]>(
    () =>
      groups.flatMap((group) => [
        { kind: 'heading' as const, key: `heading-${group.workspace}`, workspace: group.workspace },
        ...group.models.map((model) => {
          const urn = getURNFromNamedEntityRef(model) ?? `${group.workspace}/${model.name}`;
          return { kind: 'model' as const, key: urn, model, urn };
        }),
      ]),
    [groups]
  );

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: (index) => (rows[index]?.kind === 'heading' ? HEADING_HEIGHT : ITEM_HEIGHT),
    overscan: OVERSCAN,
    initialRect: INITIAL_RECT,
  });

  const virtualItems = virtualizer.getVirtualItems();
  const lastVisibleIndex =
    virtualItems.length > 0 ? virtualItems[virtualItems.length - 1].index : -1;

  const loadMore = useCallback(async () => {
    if (!onLoadMore || !hasMore || isLoadingMore || isLoadingMoreLocal) return;
    setIsLoadingMoreLocal(true);
    try {
      await onLoadMore();
    } finally {
      setIsLoadingMoreLocal(false);
    }
  }, [hasMore, isLoadingMore, isLoadingMoreLocal, onLoadMore]);

  useEffect(() => {
    if (lastVisibleIndex < 0) return;
    if (lastVisibleIndex >= rows.length - LOAD_MORE_THRESHOLD) {
      void loadMore();
    }
  }, [lastVisibleIndex, loadMore, rows.length]);

  const loadingMore = isLoadingMore || isLoadingMoreLocal;
  const showDoneMessage = Boolean(doneLoadingMessage) && !hasMore && !loading && !loadingMore;

  const renderRow = (row: ModelRow) =>
    row.kind === 'heading' ? (
      <DropdownHeading>
        <Flex gap="density-sm" align="center">
          {creatorToIcon(row.workspace, { className: 'text-base' })}
          <Text>{row.workspace}</Text>
        </Flex>
      </DropdownHeading>
    ) : (
      <ModelDropdownItem
        model={row.model}
        modelUrn={row.urn}
        isSelected={value?.model === row.urn}
        selectedAdapter={value?.model === row.urn ? value.adapter : undefined}
        onSelect={onSelect}
        hideAdapters={hideAdapters}
      />
    );

  if (rows.length === 0) {
    return (
      <Flex align="center" justify="center" className="w-full px-density-md py-density-lg">
        <Text className="text-secondary">
          {loading || loadingMore ? 'Loading models...' : emptyMessage}
        </Text>
      </Flex>
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
              key={rows[virtualRow.index].key}
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
          <LoaderCircle size={14} className="animate-spin" aria-hidden />
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
};
