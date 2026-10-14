// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  dispatchPreviewLine,
  formatPreviewLogsForDisplay,
  isAbortError,
  orderPreviewColumns,
} from '@studio/components/NewDataDesignerJobForm/previewApi';

describe('formatPreviewLogsForDisplay', () => {
  it('returns empty-ish input as-is', () => {
    expect(formatPreviewLogsForDisplay('')).toBe('');
    expect(formatPreviewLogsForDisplay('  ')).toBe('  ');
  });

  it('pretty-prints JSON object lines', () => {
    const input = '{"key":"value"}';
    const result = formatPreviewLogsForDisplay(input);
    expect(result).toContain('"key": "value"');
    expect(result).toContain('\n'); // indented
  });

  it('pretty-prints JSON array lines', () => {
    const input = '[1,2,3]';
    const result = formatPreviewLogsForDisplay(input);
    expect(result).toContain('[\n');
  });

  it('passes through non-JSON lines unchanged', () => {
    const input = 'plain text line';
    expect(formatPreviewLogsForDisplay(input)).toBe('plain text line');
  });

  it('handles mixed lines', () => {
    const input = 'hello\n{"a":1}\nworld';
    const result = formatPreviewLogsForDisplay(input);
    const lines = result.split('\n');
    expect(lines[0]).toBe('hello');
    expect(lines[lines.length - 1]).toBe('world');
    expect(result).toContain('"a": 1');
  });

  it('handles invalid JSON that looks like JSON', () => {
    const input = '{broken json}';
    expect(formatPreviewLogsForDisplay(input)).toBe('{broken json}');
  });
});

describe('isAbortError', () => {
  it('returns true for AbortError', () => {
    const err = new DOMException('aborted', 'AbortError');
    expect(isAbortError(err)).toBe(true);
  });

  it('returns false for other errors', () => {
    expect(isAbortError(new Error('oops'))).toBe(false);
  });

  it('returns false for non-objects', () => {
    expect(isAbortError(null)).toBe(false);
    expect(isAbortError('string')).toBe(false);
    expect(isAbortError(undefined)).toBe(false);
  });
});

describe('dispatchPreviewLine', () => {
  const makeHandlers = () => ({
    onLog: vi.fn(),
    onStatus: vi.fn(),
    onRecords: vi.fn(),
    onError: vi.fn(),
  });

  it('routes a dataset frame to onRecords without logging it', () => {
    const handlers = makeHandlers();
    dispatchPreviewLine(
      JSON.stringify({ kind: 'dataset', records: [{ question: 'Q1' }, 'not a row', null] }),
      handlers
    );

    expect(handlers.onRecords).toHaveBeenCalledWith([{ question: 'Q1' }]);
    expect(handlers.onLog).not.toHaveBeenCalled();
  });

  it('treats a dataset frame without a records list as no rows', () => {
    const handlers = makeHandlers();
    dispatchPreviewLine('{"kind":"dataset"}', handlers);

    expect(handlers.onRecords).toHaveBeenCalledWith([]);
  });

  it('logs the message of a log frame', () => {
    const handlers = makeHandlers();
    dispatchPreviewLine('{"kind":"log","level":"info","message":"Generating"}', handlers);

    expect(handlers.onLog).toHaveBeenCalledWith('Generating');
    expect(handlers.onError).not.toHaveBeenCalled();
  });

  it('reports and logs an error frame', () => {
    const handlers = makeHandlers();
    dispatchPreviewLine('{"kind":"error","message":"boom"}', handlers);

    expect(handlers.onError).toHaveBeenCalledWith('boom');
    expect(handlers.onLog).toHaveBeenCalledWith('boom');
  });

  it('logs other frames and non-JSON lines verbatim and skips blank lines', () => {
    const handlers = makeHandlers();
    dispatchPreviewLine('{"kind":"done"}', handlers);
    dispatchPreviewLine('plain text', handlers);
    dispatchPreviewLine('7', handlers);
    dispatchPreviewLine('   ', handlers);

    expect(handlers.onLog.mock.calls).toEqual([['{"kind":"done"}'], ['plain text'], ['7']]);
    expect(handlers.onRecords).not.toHaveBeenCalled();
  });

  it('reports log messages as status updates', () => {
    const handlers = makeHandlers();
    dispatchPreviewLine('{"kind":"log","level":"info","message":"Generating"}', handlers);
    dispatchPreviewLine('plain text', handlers);

    expect(handlers.onStatus.mock.calls).toEqual([['Generating'], ['plain text']]);
  });

  it('logs heartbeats without reporting them as status updates', () => {
    const handlers = makeHandlers();
    dispatchPreviewLine('{"kind":"heartbeat"}', handlers);

    expect(handlers.onLog).toHaveBeenCalledWith('{"kind":"heartbeat"}');
    expect(handlers.onStatus).not.toHaveBeenCalled();
  });

  it('pretty-prints JSON status updates', () => {
    const handlers = makeHandlers();
    dispatchPreviewLine('{"kind":"done"}', handlers);
    dispatchPreviewLine('{"kind":"log","message":"{\\"step\\":3}"}', handlers);

    expect(handlers.onStatus.mock.calls).toEqual([
      ['{\n  "kind": "done"\n}'],
      ['{\n  "step": 3\n}'],
    ]);
  });
});

describe('orderPreviewColumns', () => {
  it('puts config columns first in config order, then unknown keys in stream order', () => {
    expect(
      orderPreviewColumns([{ b: 1, extra: 2, a: 3 }, { c: 4 }], ['a', 'missing', 'b'])
    ).toEqual(['a', 'b', 'extra', 'c']);
  });
});
