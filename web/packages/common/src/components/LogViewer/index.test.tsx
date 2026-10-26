// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LogViewer } from '@nemo/common/src/components/LogViewer';
import type { HelixJobLog } from '@nemo/sdk/generated/platform/schema';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Shiki loads asynchronously; highlighting isn't under test.
vi.mock('@nemo/common/src/components/LogViewer/extensions/useShikiHighlighting', () => ({
  useShikiHighlighting: () => [],
}));

function makeLog(index: number): HelixJobLog {
  return {
    timestamp: `2026-01-01T00:00:0${index}Z`,
    job: 'test-job',
    job_step: 'step',
    job_task: 'task',
    message: `Log message ${index}`,
  };
}

describe('LogViewer', () => {
  it('shows a bare spinner while loading with no progress reported', () => {
    render(<LogViewer logs={[]} isLoading />);

    expect(screen.getByLabelText('Loading...')).toBeInTheDocument();
    expect(screen.queryByText(/Loading logs\.\.\./)).not.toBeInTheDocument();
  });

  it('narrates the page walk when progress is reported', () => {
    render(<LogViewer logs={[]} isLoading loadProgress={{ loaded: 12000, total: 40132 }} />);

    // Thousands separators: five-digit line counts are the whole point.
    expect(screen.getByText('Loading logs... 12,000 of 40,132 lines')).toBeInTheDocument();
    expect(screen.getByLabelText('Loading...')).toBeInTheDocument();
  });

  it('says it is loading before the first page establishes a total', () => {
    render(<LogViewer logs={[]} isLoading loadProgress={{ loaded: 0, total: 0 }} />);

    // The denominator is unknown until the first page lands, but the wait starts now.
    expect(screen.getByText('Loading logs...')).toBeInTheDocument();
  });

  it('drops the progress caption once logs are on screen', () => {
    render(<LogViewer logs={[makeLog(0), makeLog(1)]} loadProgress={{ loaded: 2, total: 2 }} />);

    expect(screen.queryByText(/Loading logs\.\.\./)).not.toBeInTheDocument();
    expect(screen.getByText('2 lines')).toBeInTheDocument();
  });

  it('mounts the live region before the first page establishes a total', () => {
    // A live region that appears already populated is skipped by screen readers that
    // only announce mutations inside a region already in the DOM, so it has to be
    // present (and empty) for the whole load.
    const { rerender } = render(<LogViewer logs={[]} isLoading loadProgress={null} />);

    const region = screen.getByTestId('log-load-progress');
    expect(region).toHaveAttribute('aria-live', 'polite');
    expect(region.textContent).toBe('Loading logs...');

    rerender(<LogViewer logs={[]} isLoading loadProgress={{ loaded: 1000, total: 40132 }} />);

    expect(region.textContent).toBe('Loading logs... 1,000 of 40,132 lines');
  });

  it('coalesces rapid page reports into one announcement per interval', () => {
    vi.useFakeTimers();
    try {
      const { rerender } = render(
        <LogViewer logs={[]} isLoading loadProgress={{ loaded: 1000, total: 40132 }} />
      );
      const region = screen.getByTestId('log-load-progress');

      // Three more pages land back to back. The first change goes out immediately;
      // the rest must collapse into a single trailing update rather than queueing a
      // polite announcement per page.
      for (const loaded of [2000, 3000, 4000]) {
        rerender(<LogViewer logs={[]} isLoading loadProgress={{ loaded, total: 40132 }} />);
      }
      expect(region.textContent).toBe('Loading logs... 2,000 of 40,132 lines');

      act(() => vi.advanceTimersByTime(1_000));

      expect(region.textContent).toBe('Loading logs... 4,000 of 40,132 lines');
    } finally {
      vi.useRealTimers();
    }
  });

  it('groups thousands in the line count', () => {
    const logs = Array.from({ length: 1200 }, (_, i) => makeLog(i));

    render(<LogViewer logs={logs} rows={30} />);

    expect(screen.getByText(/30 of 1,200 lines/)).toBeInTheDocument();
  });

  it('renders the empty message when there are no logs and nothing is loading', () => {
    render(<LogViewer logs={[]} emptyMessage="No logs for this job." />);

    expect(screen.getByText('No logs for this job.')).toBeInTheDocument();
  });

  it('shows the tail of the log in the editor', () => {
    const logs = Array.from({ length: 5 }, (_, i) => makeLog(i));

    render(<LogViewer logs={logs} rows={2} />);

    const code = screen.getByTestId('log-viewer-code');
    expect(code).toHaveTextContent('Log message 4');
    expect(code).not.toHaveTextContent('Log message 0');
  });

  it('shows every line after loading previous logs', async () => {
    const logs = Array.from({ length: 5 }, (_, i) => makeLog(i));

    render(<LogViewer logs={logs} rows={2} />);
    await userEvent.click(screen.getByText('Load previous logs'));

    expect(screen.getByTestId('log-viewer-code')).toHaveTextContent('Log message 0');
    expect(screen.getByText('5 lines')).toBeInTheDocument();
    expect(screen.queryByText('Load previous logs')).not.toBeInTheDocument();
  });

  it('appends new lines to the editor', () => {
    const { rerender } = render(<LogViewer logs={[makeLog(0)]} fillHeight />);

    rerender(<LogViewer logs={[makeLog(0), makeLog(1)]} fillHeight />);

    const code = screen.getByTestId('log-viewer-code');
    expect(code).toHaveTextContent('Log message 0');
    expect(code).toHaveTextContent('Log message 1');
  });

  it('toggles line wrapping', async () => {
    render(<LogViewer logs={[makeLog(0)]} />);
    const wrap = screen.getByRole('button', { name: 'Wrap lines' });

    await userEvent.click(wrap);

    expect(wrap).toHaveAttribute('aria-pressed', 'true');
  });
});
