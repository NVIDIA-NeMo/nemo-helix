// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parsePreviewFrame } from '@studio/routes/AnonymizerBuilderRoute/previewApi';

describe('parsePreviewFrame', () => {
  it('reads log frames', () => {
    expect(parsePreviewFrame('{"kind":"log","level":"warning","message":"slow"}')).toEqual({
      kind: 'log',
      level: 'warning',
      message: 'slow',
    });
  });

  it('defaults an unknown log level to info', () => {
    expect(parsePreviewFrame('{"kind":"log","level":"trace","message":"x"}')).toEqual({
      kind: 'log',
      level: 'info',
      message: 'x',
    });
  });

  it('reads the trace dataset and its text column', () => {
    const line = '{"kind":"trace_dataset","records":[{"a":1}],"original_text_column":"biography"}';

    expect(parsePreviewFrame(line)).toEqual({
      kind: 'trace_dataset',
      records: [{ a: 1 }],
      original_text_column: 'biography',
    });
  });

  it('leaves the text column undefined when the server omits it', () => {
    expect(parsePreviewFrame('{"kind":"trace_dataset","records":[]}')).toEqual({
      kind: 'trace_dataset',
      records: [],
      original_text_column: undefined,
    });
  });

  it('drops record entries that are not objects', () => {
    expect(parsePreviewFrame('{"kind":"failed_records","records":[1,{"b":2}]}')).toEqual({
      kind: 'failed_records',
      records: [{ b: 2 }],
    });
  });

  it('reads the control frames', () => {
    expect(parsePreviewFrame('{"kind":"done"}')).toEqual({ kind: 'done' });
    expect(parsePreviewFrame('{"kind":"heartbeat"}')).toEqual({ kind: 'heartbeat' });
    expect(parsePreviewFrame('{"kind":"error","message":"boom"}')).toEqual({
      kind: 'error',
      message: 'boom',
    });
  });

  it('reads model_download frames for both statuses', () => {
    expect(
      parsePreviewFrame('{"kind":"model_download","status":"started","message":"Downloading…"}')
    ).toEqual({
      kind: 'model_download',
      status: 'started',
      message: 'Downloading…',
    });
    expect(
      parsePreviewFrame('{"kind":"model_download","status":"complete","message":"ready"}')
    ).toEqual({
      kind: 'model_download',
      status: 'complete',
      message: 'ready',
    });
  });

  it('drops a model_download frame with an unknown status', () => {
    expect(
      parsePreviewFrame('{"kind":"model_download","status":"bogus","message":"x"}')
    ).toBeUndefined();
  });

  it('ignores blank lines, malformed JSON, and unknown frame kinds', () => {
    expect(parsePreviewFrame('  ')).toBeUndefined();
    expect(parsePreviewFrame('{not json')).toBeUndefined();
    expect(parsePreviewFrame('[1,2]')).toBeUndefined();
    expect(parsePreviewFrame('{"kind":"something_new"}')).toBeUndefined();
    expect(parsePreviewFrame('{"records":[]}')).toBeUndefined();
  });
});
