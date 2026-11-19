// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatToolPayload } from '@nemo/common/src/components/AssistantChat/work/workParts';

describe('formatToolPayload', () => {
  it('pretty-prints JSON values and JSON strings, and keeps other text as is', () => {
    expect(formatToolPayload({ a: 1 })).toEqual({ language: 'json', text: '{\n  "a": 1\n}' });
    expect(formatToolPayload('{"a":1}')).toEqual({ language: 'json', text: '{\n  "a": 1\n}' });
    expect(formatToolPayload('plain output')).toEqual({ language: 'text', text: 'plain output' });
  });

  it.each(['{"id":12345678901234567890}', '{"score":1.0}', '{"n":1e2}'])(
    'keeps %s as sent when parsing would change a number',
    (payload) => {
      expect(formatToolPayload(payload)).toEqual({ language: 'json', text: payload });
    }
  );

  it('pretty-prints JSON whose numbers survive parsing', () => {
    expect(formatToolPayload('{"id":42,"ratio":-0.5,"tag":"v2"}')).toEqual({
      language: 'json',
      text: '{\n  "id": 42,\n  "ratio": -0.5,\n  "tag": "v2"\n}',
    });
  });
});
