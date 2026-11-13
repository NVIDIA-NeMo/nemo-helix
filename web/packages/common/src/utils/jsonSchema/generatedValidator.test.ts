// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  buildMetaSchemaValidatorSources,
  DTS_PATH,
  JS_PATH,
} from '@nemo/common/scripts/metaSchemaValidatorSources';
import { readFileSync } from 'node:fs';

describe('generated meta-schema validator', () => {
  it('is in sync with the generator', () => {
    // The artifact is committed, so an ajv bump (or an edit to the generator)
    // would otherwise leave a stale validator in the tree with nothing to
    // catch it. Regenerate with:
    //   pnpm --filter @nemo/common gen:meta-schema-validator
    const { js, dts } = buildMetaSchemaValidatorSources();
    expect(readFileSync(JS_PATH, 'utf8')).toBe(js);
    expect(readFileSync(DTS_PATH, 'utf8')).toBe(dts);
  });

  it('contains no runtime code generation', () => {
    // The whole point: Studio's CSP has no 'unsafe-eval', so `new Function`
    // or `eval` anywhere in this module would throw in a deployed Studio and
    // take down the dataset schema editor. Also catches a stray CJS
    // `require()` that ajv's ESM output hard-codes for its runtime helpers,
    // which would fail to load in a browser.
    const source = readFileSync(JS_PATH, 'utf8');
    expect(source).not.toMatch(/new Function\s*\(/);
    expect(source).not.toMatch(/\beval\s*\(/);
    expect(source).not.toMatch(/\brequire\s*\(/);
  });
});
