// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatToolPayload } from '@nemo/common/src/components/AssistantChat/work/workParts';

describe('formatToolPayload', () => {
  it('pretty-prints JSON values and JSON strings, and keeps other text as is', () => {
    expect(formatToolPayload({ a: 1 })).toEqual({ language: 'json', text: '{\n  "a": 1\n}' });
    expect(formatToolPayload('{"a":1}')).toEqual({ language: 'json', text: '{\n  "a": 1\n}' });
    expect(formatToolPayload('plain output')).toEqual({ language: 'text', text: 'plain output' });
  });
});
