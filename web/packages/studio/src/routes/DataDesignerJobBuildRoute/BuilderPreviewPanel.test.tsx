// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BuilderPreviewPanel } from '@studio/routes/DataDesignerJobBuildRoute/BuilderPreviewPanel';
import { act, fireEvent, render, screen } from '@testing-library/react';

const renderPanel = ({
  isExpanded = true,
  isPreviewing = false,
  onToggleExpanded = () => undefined,
}: { isExpanded?: boolean; isPreviewing?: boolean; onToggleExpanded?: () => void } = {}) =>
  render(
    <BuilderPreviewPanel
      records={[{ topic: 'invoice' }]}
      columnOrder={['topic']}
      logs="Sampling topics"
      status={undefined}
      isPreviewing={isPreviewing}
      error={undefined}
      isExpanded={isExpanded}
      onToggleExpanded={onToggleExpanded}
    />
  );

describe('BuilderPreviewPanel', () => {
  beforeEach(() => {
    window.innerHeight = 1000;
  });

  it('shows both tabs, the rows and a collapse button when expanded', () => {
    renderPanel();

    expect(screen.getByRole('tab', { name: 'Preview Rows' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Preview Run Logs' })).toBeInTheDocument();
    expect(screen.getByRole('table')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Collapse preview' })).toBeInTheDocument();
  });

  it('shows a Preview Rows title instead of the tabs when collapsed', () => {
    renderPanel({ isExpanded: false });

    expect(screen.getByText('Preview Rows')).toBeInTheDocument();
    expect(screen.queryByRole('tab')).not.toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
    expect(screen.queryByRole('separator')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Generating preview rows')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Expand preview' })).toBeInTheDocument();
  });

  it('shows a spinner next to the collapsed title while the preview runs', () => {
    renderPanel({ isExpanded: false, isPreviewing: true });

    expect(screen.getByLabelText('Generating preview rows')).toBeInTheDocument();
  });

  it('calls onToggleExpanded when the chevron is clicked', () => {
    const onToggleExpanded = vi.fn();
    renderPanel({ onToggleExpanded });

    fireEvent.click(screen.getByRole('button', { name: 'Collapse preview' }));

    expect(onToggleExpanded).toHaveBeenCalledTimes(1);
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

  it('pulls the panel back within the limit when the window shrinks', () => {
    renderPanel();
    const handle = screen.getByRole('separator', { name: 'Resize preview panel' });
    fireEvent.keyDown(handle, { key: 'Home' });
    expect(handle).toHaveAttribute('aria-valuenow', '800');

    act(() => {
      window.innerHeight = 500;
      window.dispatchEvent(new Event('resize'));
    });

    expect(handle).toHaveAttribute('aria-valuemax', '400');
    expect(handle).toHaveAttribute('aria-valuenow', '400');
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
