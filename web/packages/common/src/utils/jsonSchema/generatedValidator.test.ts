// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  buildMetaSchemaValidatorSources,
  DTS_PATH,
  JS_PATH,
} from '@nemo/common/scripts/metaSchemaValidatorSources';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

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
    // `require()`, which would fail to load in a browser.
    const source = readFileSync(JS_PATH, 'utf8');
    expect(source).not.toMatch(/new Function\s*\(/);
    expect(source).not.toMatch(/\beval\s*\(/);
    expect(source).not.toMatch(/\brequire\s*\(/);
  });

  it('runs under native node ESM', () => {
    // Vitest loads this module through Vite, whose esbuild interop silently
    // repairs a default import of a CommonJS module. Node's own ESM loader
    // does not: it binds the import to `module.exports`, so ajv's runtime
    // helpers arrive as `{default: fn}` rather than `fn` and throw when
    // called. Only a real node process exercises that, and only an input
    // that reaches a helper — `uniqueItems` on an array-valued `type` is the
    // one path into ajv's `equal` here, which is why validating an ordinary
    // object schema sails past it.
    const script = [
      `import { validate } from ${JSON.stringify(resolve(JS_PATH))};`,
      `const ok = validate({ type: ['string', 'number'] });`,
      `const dupe = validate({ type: ['string', 'string'] });`,
      `const plain = validate({ type: 'object', properties: { a: { type: 'string' } } });`,
      `const bad = validate({ type: 'strng' });`,
      `console.log(JSON.stringify({ ok, dupe, plain, bad }));`,
    ].join('\n');

    const out = execFileSync(process.execPath, ['--input-type=module', '--eval', script], {
      cwd: resolve('.'),
      encoding: 'utf8',
    });

    expect(JSON.parse(out.trim())).toEqual({
      ok: true, // distinct types: valid
      dupe: false, // duplicate entries violate uniqueItems
      plain: true,
      bad: false,
    });
  });
});
