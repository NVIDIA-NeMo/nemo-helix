// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EditorView } from '@codemirror/view';
import { LogViewer } from '@nemo/common/src/components/LogViewer';
import type { HelixJobLog } from '@nemo/sdk/generated/platform/schema';
import { act, fireEvent, render, screen } from '@testing-library/react';
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

  it.each([
    ['live', 'Live'],
    ['ended', 'Ended'],
  ] as const)('badges a %s log as %s', (logStatus, label) => {
    render(<LogViewer logs={[makeLog(0)]} logStatus={logStatus} />);

    expect(screen.getByTestId('nv-badge')).toHaveTextContent(label);
  });

  it('shows no badge when the log status is unknown', () => {
    render(<LogViewer logs={[makeLog(0)]} />);

    expect(screen.queryByTestId('nv-badge')).not.toBeInTheDocument();
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

  describe('syncing the document', () => {
    const getView = (): EditorView => {
      const view = EditorView.findFromDOM(screen.getByTestId('log-viewer-code'));
      if (!view) throw new Error('editor not mounted');
      return view;
    };
    const range = (from: number, to: number) =>
      Array.from({ length: to - from }, (_, i) => makeLog(from + i));
    const docLines = () => getView().state.doc.toString().split('\n');
    // jsdom has no layout, so the editor always looks scrolled to the bottom.
    const scrollUp = () => {
      const scroller = getView().scrollDOM;
      Object.defineProperty(scroller, 'scrollHeight', { configurable: true, value: 1000 });
      Object.defineProperty(scroller, 'clientHeight', { configurable: true, value: 100 });
      fireEvent.scroll(scroller);
    };
    // @uiw/react-codemirror also dispatches its own reconfigures on rerender.
    const spyOnDocChanges = () => {
      const dispatch = vi.spyOn(getView(), 'dispatch');
      return () =>
        dispatch.mock.calls
          .map(([spec]) => spec)
          .filter((spec) => typeof spec === 'object' && 'changes' in spec);
    };

    it('applies a sliding window as a head trim plus a tail append', () => {
      const { rerender } = render(<LogViewer logs={range(0, 5)} fillHeight />);
      scrollUp();
      const docChanges = spyOnDocChanges();

      rerender(<LogViewer logs={range(2, 8)} fillHeight />);

      // Lines 2-4 are never rewritten, which is what keeps CodeMirror's scroll anchor.
      const line = (i: number) => `[${makeLog(i).timestamp}]   Log message ${i}`;
      const oldLength = range(0, 5)
        .map((_, i) => line(i))
        .join('\n').length;
      expect(docChanges()).toEqual([
        {
          changes: [
            { from: 0, to: `${line(0)}\n${line(1)}\n`.length },
            { from: oldLength, insert: `\n${line(5)}\n${line(6)}\n${line(7)}` },
          ],
        },
      ]);
      expect(docLines()).toEqual([2, 3, 4, 5, 6, 7].map(line));
    });

    it('slides the tail without replacing it', () => {
      const { rerender } = render(<LogViewer logs={range(0, 5)} rows={2} />);
      scrollUp();
      const docChanges = spyOnDocChanges();

      rerender(<LogViewer logs={range(0, 6)} rows={2} />);

      expect(docChanges()).toEqual([
        {
          changes: [
            { from: 0, to: expect.any(Number) },
            { from: expect.any(Number), insert: `\n[${makeLog(5).timestamp}]   Log message 5` },
          ],
        },
      ]);
      expect(docLines()).toEqual([4, 5].map((i) => `[${makeLog(i).timestamp}]   Log message ${i}`));
    });

    it('counts multi-line and carriage-return messages when trimming', () => {
      const traceback = { ...makeLog(0), message: 'Traceback\r\n  File "a.py"\n  boom' };
      const progress = { ...makeLog(1), message: 'progress 50%\r' };
      const { rerender } = render(
        <LogViewer logs={[traceback, progress, makeLog(2)]} fillHeight />
      );
      scrollUp();

      rerender(<LogViewer logs={[progress, makeLog(2), makeLog(3)]} fillHeight />);

      expect(docLines()).toEqual([
        `[${progress.timestamp}]   progress 50%`,
        '',
        `[${makeLog(2).timestamp}]   Log message 2`,
        `[${makeLog(3).timestamp}]   Log message 3`,
      ]);
    });

    it('replaces the document for a reader following the bottom when lines are trimmed', () => {
      const { rerender } = render(<LogViewer logs={range(0, 5)} fillHeight />);
      const docChanges = spyOnDocChanges();

      rerender(<LogViewer logs={range(2, 6)} fillHeight />);

      expect(docChanges()).toEqual([
        expect.objectContaining({ changes: expect.objectContaining({ from: 0 }) }),
      ]);
      expect(docLines()).toHaveLength(4);
      expect(docLines()[0]).toBe(`[${makeLog(2).timestamp}]   Log message 2`);
    });

    it('appends for a reader following the bottom', () => {
      const { rerender } = render(<LogViewer logs={range(0, 2)} fillHeight />);
      const docChanges = spyOnDocChanges();

      rerender(<LogViewer logs={range(0, 3)} fillHeight />);

      expect(docChanges()).toEqual([
        {
          changes: [
            { from: expect.any(Number), insert: `\n[${makeLog(2).timestamp}]   Log message 2` },
          ],
        },
      ]);
    });

    it('keeps no undo history, so undo shortcuts cannot revert synced logs', () => {
      const { rerender } = render(<LogViewer logs={range(0, 2)} fillHeight />);
      rerender(<LogViewer logs={range(1, 4)} fillHeight />);
      const expected = docLines();

      const content = getView().contentDOM;
      fireEvent.keyDown(content, { key: 'z', ctrlKey: true });
      fireEvent.keyDown(content, { key: 'u', ctrlKey: true });

      expect(docLines()).toEqual(expected);
    });

    it('strips ANSI colour codes from what it shows', () => {
      const colored = { ...makeLog(0), message: '\x1b[2mdim\x1b[0m [\x1b[32minfo\x1b[0m] done' };

      render(<LogViewer logs={[colored]} fillHeight />);

      expect(docLines()).toEqual([`[${colored.timestamp}]   dim [info] done`]);
    });

    it('copies the text it shows, without ANSI codes or carriage returns', async () => {
      // `userEvent.setup()` stubs `navigator.clipboard`.
      const user = userEvent.setup();
      const colored = { ...makeLog(0), message: '\x1b[32mok\x1b[0m\r\nnext' };
      render(<LogViewer logs={[colored]} fillHeight onNotify={vi.fn()} />);

      await user.click(screen.getByRole('button', { name: 'Copy logs' }));

      expect(await navigator.clipboard.readText()).toBe(docLines().join('\n'));
      expect(docLines()).toEqual([`[${colored.timestamp}]   ok`, 'next']);
    });

    it('leaves keyboard navigation to the browser instead of moving a hidden cursor', () => {
      render(<LogViewer logs={range(0, 3)} fillHeight />);
      const view = getView();
      const head = view.state.selection.main.head;

      fireEvent.keyDown(view.contentDOM, { key: 'End' });
      fireEvent.keyDown(view.contentDOM, { key: 'ArrowDown' });

      expect(view.state.selection.main.head).toBe(head);
    });

    it('keeps a scroll-locked tail following the bottom when something else scrolls it', () => {
      const { rerender } = render(<LogViewer logs={range(0, 5)} rows={2} />);
      // CodeMirror's search can still scroll the tail that the CSS locks against the reader.
      getView().scrollDOM.style.overflowY = 'hidden';
      scrollUp();
      const docChanges = spyOnDocChanges();

      rerender(<LogViewer logs={range(0, 6)} rows={2} />);

      // Still pinned, so the slid tail is replaced and re-pinned rather than edited in place.
      expect(docChanges()).toEqual([
        expect.objectContaining({ changes: expect.objectContaining({ from: 0 }) }),
      ]);
    });

    it('replaces the document when an earlier line changes', () => {
      const { rerender } = render(<LogViewer logs={range(0, 3)} fillHeight />);
      const rewritten = [makeLog(0), { ...makeLog(1), message: 'rewritten' }, makeLog(2)];

      rerender(<LogViewer logs={rewritten} fillHeight />);

      expect(docLines()[1]).toBe(`[${makeLog(1).timestamp}]   rewritten`);
      expect(docLines()).toHaveLength(3);
    });

    it('leaves the document alone when the entries are unchanged', () => {
      const { rerender } = render(<LogViewer logs={range(0, 3)} fillHeight />);
      const docChanges = spyOnDocChanges();

      rerender(<LogViewer logs={range(0, 3)} fillHeight />);

      expect(docChanges()).toEqual([]);
    });
  });

  it('names the focusable log content for assistive technology', () => {
    render(<LogViewer logs={[makeLog(0)]} />);

    expect(screen.getByRole('textbox', { name: 'Logs' })).toHaveAttribute('aria-readonly', 'true');
  });

  it('caps the tail on the shrinkable wrapper so a short slot cannot clip the newest lines', () => {
    render(<LogViewer logs={[makeLog(0)]} rows={5} />);

    expect(screen.getByTestId('log-viewer-code')).toHaveClass('log-viewer--capped');
  });

  it('does not cap a log that fills its container', () => {
    render(<LogViewer logs={[makeLog(0)]} fillHeight />);

    expect(screen.getByTestId('log-viewer-code')).not.toHaveClass('log-viewer--capped');
  });

  it('toggles line wrapping', async () => {
    render(<LogViewer logs={[makeLog(0)]} />);
    const wrap = screen.getByRole('button', { name: 'Wrap lines' });

    await userEvent.click(wrap);

    expect(wrap).toHaveAttribute('aria-pressed', 'true');
  });
});
