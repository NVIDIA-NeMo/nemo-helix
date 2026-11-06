// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { VirtualDropdownList } from '@nemo/common/src/components/VirtualDropdownList';
import { render, waitFor } from '@testing-library/react';
import { useState } from 'react';

const rows = [{ key: 'a' }, { key: 'b' }];
const TOTAL_PAGES = 3;

/** Every page after the first is filtered to nothing, so rows never grow while pages remain. */
const FilteredPages = ({ onLoadMore }: { onLoadMore: () => Promise<void> }) => {
  const [page, setPage] = useState(1);
  return (
    <VirtualDropdownList
      rows={rows}
      renderRow={(row) => row.key}
      estimateSize={() => 36}
      onLoadMore={async () => {
        await onLoadMore();
        setPage((p) => p + 1);
      }}
      hasMore={page < TOTAL_PAGES}
      emptyMessage="None"
      loadingMessage="Loading"
    />
  );
};

describe('VirtualDropdownList paging', () => {
  // jsdom has no layout; give the list a viewport and rows a height so the virtualizer windows.
  beforeEach(() => {
    vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(300);
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      width: 360,
      height: 36,
    } as DOMRect);
  });

  it('keeps paging when a loaded page adds no rows', async () => {
    const onLoadMore = vi.fn().mockResolvedValue(undefined);
    render(<FilteredPages onLoadMore={onLoadMore} />);

    await waitFor(() => expect(onLoadMore).toHaveBeenCalledTimes(TOTAL_PAGES - 1));
  });

  it('does not retry a page that failed to load', async () => {
    const onLoadMore = vi.fn().mockRejectedValue(new Error('boom'));
    render(<FilteredPages onLoadMore={onLoadMore} />);

    await waitFor(() => expect(onLoadMore).toHaveBeenCalledTimes(1));
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(onLoadMore).toHaveBeenCalledTimes(1);
  });
});
