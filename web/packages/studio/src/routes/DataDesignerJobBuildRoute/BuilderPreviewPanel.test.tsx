// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BuilderPreviewPanel } from '@studio/routes/DataDesignerJobBuildRoute/BuilderPreviewPanel';
import { fireEvent, render, screen } from '@testing-library/react';

const renderPanel = (onClose: () => void = () => undefined) =>
  render(
    <BuilderPreviewPanel
      records={[{ topic: 'invoice' }]}
      columnOrder={['topic']}
      logs="Sampling topics"
      isPreviewing={false}
      error={undefined}
      onClose={onClose}
    />
  );

describe('BuilderPreviewPanel', () => {
  beforeEach(() => {
    window.innerHeight = 1000;
  });

  it('shows both tabs and a close button', () => {
    renderPanel();

    expect(screen.getByRole('tab', { name: 'Preview Rows' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Preview Run Logs' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Close preview' })).toBeInTheDocument();
  });

  it('calls onClose when the close button is clicked', () => {
    const onClose = vi.fn();
    renderPanel(onClose);

    fireEvent.click(screen.getByRole('button', { name: 'Close preview' }));

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('grows when the handle is dragged up and shrinks when dragged down', () => {
    renderPanel();
    const handle = screen.getByRole('separator', { name: 'Resize preview panel' });
    const initial = Number(handle.getAttribute('aria-valuenow'));

    fireEvent.mouseDown(handle, { clientY: 600 });
    fireEvent.mouseMove(window, { clientY: 500 });
    expect(handle).toHaveAttribute('aria-valuenow', String(initial + 100));

    fireEvent.mouseMove(window, { clientY: 650 });
    expect(handle).toHaveAttribute('aria-valuenow', String(initial - 50));
    fireEvent.mouseUp(window);
  });

  it('stays within its minimum and maximum height', () => {
    renderPanel();
    const handle = screen.getByRole('separator', { name: 'Resize preview panel' });
    const min = Number(handle.getAttribute('aria-valuemin'));
    const max = Number(handle.getAttribute('aria-valuemax'));

    fireEvent.mouseDown(handle, { clientY: 600 });
    fireEvent.mouseMove(window, { clientY: -5000 });
    expect(handle).toHaveAttribute('aria-valuenow', String(max));
    fireEvent.mouseMove(window, { clientY: 5000 });
    expect(handle).toHaveAttribute('aria-valuenow', String(min));
    fireEvent.mouseUp(window);
  });

  it('resizes with the arrow, Home and End keys', () => {
    renderPanel();
    const handle = screen.getByRole('separator', { name: 'Resize preview panel' });
    const initial = Number(handle.getAttribute('aria-valuenow'));

    fireEvent.keyDown(handle, { key: 'ArrowUp' });
    expect(Number(handle.getAttribute('aria-valuenow'))).toBeGreaterThan(initial);

    fireEvent.keyDown(handle, { key: 'End' });
    expect(handle).toHaveAttribute('aria-valuenow', handle.getAttribute('aria-valuemin'));

    fireEvent.keyDown(handle, { key: 'Home' });
    expect(handle).toHaveAttribute('aria-valuenow', handle.getAttribute('aria-valuemax'));
  });
});
