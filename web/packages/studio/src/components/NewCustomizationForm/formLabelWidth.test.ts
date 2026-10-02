// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Both label columns in this form are a fixed width, and both clip what does not fit:
 * KUI gives `labelPosition="left"` a 160px column (`--nv-form-field-label-group-width`)
 * and `SliderWithTextInput` hardcodes `w-[165px]`, each rendering the label with
 * `overflow: hidden; text-overflow: ellipsis; white-space: nowrap`. A label wider than its
 * column is therefore clipped at every viewport size — resizing the window never reveals
 * it. An info popover costs a further 16px of that column.
 *
 * What actually matters is rendered width, and these budgets are character counts, which
 * is a poor stand-in for a proportional typeface: measured in Chromium, the slider label
 * "Min Report Interval (s)" is 23 characters and fits at 147px, while "Label Smoothing
 * Factor" is 22 and was clipped at 156px. The budgets below are therefore set where they
 * flag no label that currently fits, which means they catch a plainly over-long addition
 * and will miss a short-but-wide one. Measure before trusting them either way, and do not
 * raise one to make a new label pass without measuring it first.
 */
const MAX_LABEL_CHARS = { switch: 22, slider: 23 } as const;

const FORM_DIR = dirname(fileURLToPath(import.meta.url));

interface Label {
  file: string;
  label: string;
  kind: keyof typeof MAX_LABEL_CHARS;
}

const fixedColumnLabels = (): Label[] =>
  readdirSync(FORM_DIR)
    .filter((f) => f.endsWith('.tsx') && !f.includes('.test.'))
    .flatMap((file) =>
      readFileSync(join(FORM_DIR, file), 'utf8')
        .split(/(?=<Controlled)/)
        .flatMap((chunk): Label[] => {
          const label = chunk.match(/slotLabel:\s*'((?:[^'\\]|\\.)*)'/)?.[1];
          if (!label) return [];
          if (chunk.startsWith('<ControlledSliderWithTextInput'))
            return [{ file, label, kind: 'slider' as const }];
          if (chunk.includes("labelPosition: 'left'"))
            return [{ file, label, kind: 'switch' as const }];
          return [];
        })
    );

describe('labels in a fixed-width column', () => {
  it('finds the labels it is meant to be guarding', () => {
    const found = fixedColumnLabels();
    expect(found.filter((l) => l.kind === 'slider').length).toBeGreaterThan(50);
    expect(found.filter((l) => l.kind === 'switch').length).toBeGreaterThan(10);
  });

  it('stay within their column', () => {
    const tooLong = fixedColumnLabels()
      .filter(({ label, kind }) => label.length > MAX_LABEL_CHARS[kind])
      .map(({ file, label, kind }) => `${kind} "${label}" (${label.length} chars) in ${file}`);

    expect(tooLong).toEqual([]);
  });
});
