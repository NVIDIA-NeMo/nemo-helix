// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ResizeablePanel } from '@studio/components/common/ResizeablePanel';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

describe('ResizeablePanel', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('resizes both directions from the keyboard', async () => {
    const user = userEvent.setup();
    render(
      <ResizeablePanel
        slotLeft="Graph"
        slotRight="Span details"
        defaultLeftWidth={680}
        minLeftWidth={480}
        maxLeftWidth={900}
      />
    );
    const separator = screen.getByRole('separator', { name: 'Resize panels' });

    separator.focus();
    await user.keyboard('{ArrowRight}');
    expect(separator).toHaveAttribute('aria-valuenow', '704');

    await user.keyboard('{Home}');
    expect(separator).toHaveAttribute('aria-valuenow', '480');
  });

  it('stacks the panels when both minimum widths do not fit', () => {
    let notifyResize!: ResizeObserverCallback;
    vi.stubGlobal(
      'ResizeObserver',
      class {
        constructor(callback: ResizeObserverCallback) {
          notifyResize = callback;
        }
        observe() {}
        disconnect() {}
      }
    );

    render(
      <ResizeablePanel
        slotLeft="Graph"
        slotRight="Span details"
        minLeftWidth={480}
        minRightWidth={352}
      />
    );

    act(() =>
      notifyResize([{ contentRect: { width: 800 } } as ResizeObserverEntry], {} as ResizeObserver)
    );
    expect(screen.queryByRole('separator', { name: 'Resize panels' })).not.toBeInTheDocument();

    act(() =>
      notifyResize([{ contentRect: { width: 1000 } } as ResizeObserverEntry], {} as ResizeObserver)
    );
    expect(screen.getByRole('separator', { name: 'Resize panels' })).toBeInTheDocument();
  });
});
