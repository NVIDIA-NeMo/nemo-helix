// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Every `specSliderProps` call has to name the field its own control is bound to, and read
 * from its own backend's table.
 *
 * This is checked by reading the source because nothing else can see it. A field name that
 * belongs to a different control is still a valid string and still resolves — to another
 * field's value, or to `undefined` — so the types are satisfied, the linter is satisfied,
 * and the rendered form looks plausible. The only signal is the control's own binding.
 */

const FORM_DIR = dirname(fileURLToPath(import.meta.url));

/** `automodel.batch.micro_batch_size` → `batch_micro_batch_size`. */
const fieldForBinding = (name: string): string => {
  const [namespace, ...rest] = name.split('.');
  // `rl` binds through `training`, which is the object the RL tables are keyed off.
  return namespace === 'rl' ? rest.slice(1).join('_') : rest.join('_');
};

const tableForBinding = (name: string): string | undefined => {
  if (name.startsWith('automodel.')) return 'AUTOMODEL_SPEC_DEFAULTS';
  if (name.startsWith('unsloth.')) return 'UNSLOTH_SPEC_DEFAULTS';
  // `rl.*` and `grpo.*` read DPO or GRPO depending on which form renders the control, so
  // the arm is the section's choice rather than something the binding can tell us.
  return undefined;
};

interface SliderCall {
  file: string;
  binding: string;
  table: string;
  field: string;
}

const collectSliderCalls = (): SliderCall[] => {
  const calls: SliderCall[] = [];
  for (const file of readdirSync(FORM_DIR).filter((name) => name.endsWith('.tsx'))) {
    const source = readFileSync(join(FORM_DIR, file), 'utf8');
    // One self-closing control element at a time, so a match cannot run past a control that
    // has no defaultValue and pick up its neighbour's — which is the bug this guards.
    for (const element of source.matchAll(/<Controlled\w+\b[\s\S]*?\n\s*\/>/g)) {
      const block = element[0];
      const binding = /name: '([a-z][a-z_.0-9]+)'/.exec(block);
      const call = /specSliderProps\((\w+), '([a-z_0-9]+)'\)/.exec(block);
      if (!binding || !call) continue;
      calls.push({ file, binding: binding[1], table: call[1], field: call[2] });
    }
  }
  return calls;
};

describe('customizer slider spec fields', () => {
  const calls = collectSliderCalls();

  it('finds the slider calls to check', () => {
    expect(calls.length).toBeGreaterThan(50);
  });

  it('every slider reads the field its control is bound to', () => {
    const wrong = calls
      .filter((call) => call.field !== fieldForBinding(call.binding))
      .map((call) => `${call.file}: ${call.binding} reads '${call.field}'`);
    expect(wrong).toEqual([]);
  });

  it('every slider reads its own backend table', () => {
    const wrong = calls
      .filter((call) => {
        const expected = tableForBinding(call.binding);
        return expected !== undefined && call.table !== expected;
      })
      .map((call) => `${call.file}: ${call.binding} reads ${call.table}`);
    expect(wrong).toEqual([]);
  });
});
