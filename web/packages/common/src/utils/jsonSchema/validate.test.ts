// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseAndValidate } from '@nemo/common/src/utils/jsonSchema/validate';

describe('parseAndValidate', () => {
  it('returns the parsed object when the text is valid JSON + valid schema', () => {
    const result = parseAndValidate('{"type": "string"}');
    expect(result.valid).toBe(true);
    if (result.valid) {
      expect(result.value).toEqual({ type: 'string' });
    }
  });

  it('reports parse errors when the text is not valid JSON', () => {
    const result = parseAndValidate('{not json}');
    expect(result.valid).toBe(false);
    if (!result.valid) {
      expect(result.errors[0]).toMatch(/Invalid JSON/);
    }
  });

  it('rejects valid JSON that is not a JSON object (e.g. arrays, primitives)', () => {
    const arr = parseAndValidate('[1, 2, 3]');
    expect(arr.valid).toBe(false);
    const prim = parseAndValidate('42');
    expect(prim.valid).toBe(false);
  });
});
