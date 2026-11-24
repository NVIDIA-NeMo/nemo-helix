// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseAndValidate } from '@nemo/common/src/utils/jsonSchema';
import { validateDatasetMetadata } from '@studio/routes/FilesetDetailRoute/DatasetSchemaEditor/helpers';

// Own file so no earlier test warms a lazily compiled validator. Studio's CSP
// blocks 'unsafe-eval', so editor validation must never reach `Function`.
describe('schema editor validation under a no-eval CSP', () => {
  it('validates without evaluating strings as code', () => {
    const fn = vi.spyOn(globalThis, 'Function').mockImplementation(() => {
      throw new EvalError('unsafe-eval blocked');
    });
    try {
      const parsed = parseAndValidate('{"schema": {"type": "object"}, "schemas_by_path": {}}');
      expect(parsed.valid).toBe(true);
      if (parsed.valid) expect(validateDatasetMetadata(parsed.value).valid).toBe(true);
      expect(validateDatasetMetadata({ schemas_by_path: 5 }).valid).toBe(false);
    } finally {
      fn.mockRestore();
    }
  });
});
