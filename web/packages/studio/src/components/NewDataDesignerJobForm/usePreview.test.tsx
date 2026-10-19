// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DataDesignerConfig } from '@nemo/sdk/generated/data-designer/schema';
import type { PreviewStreamHandlers } from '@studio/components/NewDataDesignerJobForm/previewApi';
import { usePreview } from '@studio/components/NewDataDesignerJobForm/usePreview';
import { act, renderHook, waitFor } from '@testing-library/react';

const streamPreviewMock = vi.fn();
vi.mock('@studio/components/NewDataDesignerJobForm/previewApi', async () => {
  const actual = await vi.importActual<
    typeof import('@studio/components/NewDataDesignerJobForm/previewApi')
  >('@studio/components/NewDataDesignerJobForm/previewApi');
  return { ...actual, streamPreview: (...a: unknown[]) => streamPreviewMock(...a) };
});

const CONFIG = { columns: [] } as unknown as DataDesignerConfig;

describe('usePreview', () => {
  beforeEach(() => {
    streamPreviewMock.mockReset();
  });

  it('surfaces an error when building the config throws (e.g. invalid JSON field)', async () => {
    const { result } = renderHook(() =>
      usePreview({
        workspace: 'ws',
        accessToken: 'token',
        getCurrentConfig: () => {
          throw new SyntaxError('Unexpected token n in JSON at position 2');
        },
      })
    );

    await act(async () => {
      await result.current.runPreview();
    });

    // A thrown config-build error must be shown to the user, not silently swallowed.
    await waitFor(() => {
      expect(result.current.previewLogs).not.toBe('');
    });
    expect(result.current.previewLogs).toMatch(/JSON|error|preview/i);
    expect(result.current.isPreviewing).toBe(false);
    expect(streamPreviewMock).not.toHaveBeenCalled();
  });

  it('streams normally when the config builds', async () => {
    streamPreviewMock.mockImplementation(async (...args: unknown[]) => {
      (args[4] as PreviewStreamHandlers).onLog('a log line');
    });
    const { result } = renderHook(() =>
      usePreview({ workspace: 'ws', accessToken: 'token', getCurrentConfig: () => CONFIG })
    );

    await act(async () => {
      await result.current.runPreview();
    });

    expect(streamPreviewMock).toHaveBeenCalled();
    expect(result.current.previewLogs).toContain('a log line');
    expect(result.current.previewStatus).toBeUndefined();
    expect(result.current.isPreviewing).toBe(false);
    expect(result.current.hasPreviewRun).toBe(true);
    expect(result.current.previewRecords).toBeUndefined();
  });

  it('collects streamed records alongside the logs, with the config column order', async () => {
    const records = [{ question: 'Q1', scheduler: 'alice' }];
    streamPreviewMock.mockImplementation(async (...args: unknown[]) => {
      const handlers = args[4] as PreviewStreamHandlers;
      handlers.onLog('Preview generation complete');
      handlers.onRecords(records);
    });
    const config = {
      columns: [{ name: 'scheduler' }, { name: 'question' }],
    } as unknown as DataDesignerConfig;
    const { result } = renderHook(() =>
      usePreview({ workspace: 'ws', accessToken: 'token', getCurrentConfig: () => config })
    );

    await act(async () => {
      await result.current.runPreview();
    });

    expect(result.current.previewRecords).toEqual(records);
    expect(result.current.previewColumnOrder).toEqual(['scheduler', 'question']);
    expect(result.current.previewLogs).toContain('Preview generation complete');
    expect(result.current.previewError).toBeUndefined();
  });

  it('surfaces an error frame as the preview error', async () => {
    streamPreviewMock.mockImplementation(async (...args: unknown[]) => {
      (args[4] as PreviewStreamHandlers).onError('model unavailable');
    });
    const { result } = renderHook(() =>
      usePreview({ workspace: 'ws', accessToken: 'token', getCurrentConfig: () => CONFIG })
    );

    await act(async () => {
      await result.current.runPreview();
    });

    expect(result.current.previewError).toBe('model unavailable');
  });

  it('surfaces a failed request as the preview error and log line', async () => {
    streamPreviewMock.mockRejectedValue(new Error('Preview failed: 500'));
    const { result } = renderHook(() =>
      usePreview({ workspace: 'ws', accessToken: 'token', getCurrentConfig: () => CONFIG })
    );

    await act(async () => {
      await result.current.runPreview();
    });

    expect(result.current.previewError).toBe('Preview failed: 500');
    expect(result.current.previewLogs).toContain('Preview failed: 500');
  });

  describe('superseded runs', () => {
    interface OpenStream {
      handlers: PreviewStreamHandlers;
      reject: (error: Error) => void;
    }

    const openStreams = (): OpenStream[] => {
      const streams: OpenStream[] = [];
      streamPreviewMock.mockImplementation(
        (...args: unknown[]) =>
          new Promise((_resolve, reject) => {
            streams.push({ handlers: args[4] as PreviewStreamHandlers, reject });
          })
      );
      return streams;
    };

    const abortError = () => Object.assign(new Error('aborted'), { name: 'AbortError' });

    it('ignores an earlier run that reports after the next run fails to build its config', async () => {
      const streams = openStreams();
      let config: DataDesignerConfig | undefined = CONFIG;
      const { result } = renderHook(() =>
        usePreview({ workspace: 'ws', accessToken: 'token', getCurrentConfig: () => config })
      );

      let first!: Promise<void>;
      await act(async () => {
        first = result.current.runPreview();
      });
      config = undefined;
      await act(async () => {
        await result.current.runPreview();
      });

      act(() => {
        streams[0].handlers.onRecords([{ stale: 'row' }]);
        streams[0].handlers.onError('stale error');
      });
      await act(async () => {
        streams[0].reject(new Error('stale failure'));
        await first;
      });

      expect(result.current.previewRecords).toBeUndefined();
      expect(result.current.previewError).toBe('Generate a job spec first, then run Preview.');
      expect(result.current.previewLogs).not.toContain('stale failure');
      expect(result.current.isPreviewing).toBe(false);
    });

    it('groups indented status lines under the update they belong to', async () => {
      const streams = openStreams();
      const { result } = renderHook(() =>
        usePreview({ workspace: 'ws', accessToken: 'token', getCurrentConfig: () => CONFIG })
      );
      await act(async () => {
        void result.current.runPreview();
      });

      act(() => {
        streams[0].handlers.onStatus('🚀 Dispatching');
        streams[0].handlers.onStatus('📊 Progress [12.5s]:');
        streams[0].handlers.onStatus("  |-- column 'subject': 1/10 (10%)");
        streams[0].handlers.onStatus("  |-- column 'body': 0/10 (0%)");
      });
      expect(result.current.previewStatus).toBe(
        "📊 Progress [12.5s]:\n  |-- column 'subject': 1/10 (10%)\n  |-- column 'body': 0/10 (0%)"
      );

      act(() => {
        streams[0].handlers.onStatus('🔄 Salvaging 6 deferred task(s)');
      });
      expect(result.current.previewStatus).toBe('🔄 Salvaging 6 deferred task(s)');
    });

    it('keeps an earlier run’s logs and failure out of the newer run', async () => {
      const streams = openStreams();
      const { result } = renderHook(() =>
        usePreview({ workspace: 'ws', accessToken: 'token', getCurrentConfig: () => CONFIG })
      );

      let first!: Promise<void>;
      await act(async () => {
        first = result.current.runPreview();
      });
      let second!: Promise<void>;
      await act(async () => {
        second = result.current.runPreview();
      });

      act(() => {
        streams[0].handlers.onLog('stale log');
        streams[0].handlers.onStatus('stale status');
        streams[1].handlers.onLog('fresh log');
        streams[1].handlers.onStatus('fresh status');
      });
      await act(async () => {
        streams[0].reject(new Error('stale failure'));
        await first;
      });

      expect(result.current.previewLogs).toContain('fresh log');
      expect(result.current.previewLogs).not.toContain('stale log');
      expect(result.current.previewLogs).not.toContain('stale failure');
      expect(result.current.previewStatus).toBe('fresh status');
      expect(result.current.previewError).toBeUndefined();
      expect(result.current.isPreviewing).toBe(true);

      await act(async () => {
        streams[1].reject(abortError());
        await second;
      });
    });
  });

  describe('stopPreview', () => {
    /**
     * Hangs until the caller's signal aborts, standing in for a long-running stream.
     * The signal is taken positionally (streamPreview's 4th argument) — an `instanceof`
     * check can miss it when the hook's AbortController comes from another realm.
     */
    const mockHangingStream = () =>
      streamPreviewMock.mockImplementation(
        (...args: unknown[]) =>
          new Promise((_resolve, reject) => {
            const signal = args[3] as AbortSignal | undefined;
            signal?.addEventListener('abort', () => {
              const err = new Error('aborted');
              err.name = 'AbortError';
              reject(err);
            });
          })
      );

    const renderPreview = () =>
      renderHook(() =>
        usePreview({ workspace: 'ws', accessToken: 'token', getCurrentConfig: () => CONFIG })
      );

    it('aborts the in-flight run and reports the stop', async () => {
      mockHangingStream();
      const { result } = renderPreview();

      let running!: Promise<void>;
      await act(async () => {
        running = result.current.runPreview();
      });
      expect(result.current.isPreviewing).toBe(true);

      await act(async () => {
        result.current.stopPreview();
        await running;
      });

      expect(result.current.isPreviewing).toBe(false);
      expect(result.current.previewLogs).toContain('Preview stopped.');
    });

    it('is a no-op when no preview is running', () => {
      const { result } = renderPreview();
      expect(() => result.current.stopPreview()).not.toThrow();
      expect(result.current.isPreviewing).toBe(false);
    });

    it('keeps running when an older superseded run unwinds', async () => {
      mockHangingStream();
      const { result } = renderPreview();

      let first!: Promise<void>;
      await act(async () => {
        first = result.current.runPreview();
      });

      let second!: Promise<void>;
      await act(async () => {
        second = result.current.runPreview();
        await first;
      });

      expect(result.current.isPreviewing).toBe(true);
      expect(result.current.previewLogs).not.toContain('Preview stopped.');

      await act(async () => {
        result.current.stopPreview();
        await second;
      });
    });
  });
});
