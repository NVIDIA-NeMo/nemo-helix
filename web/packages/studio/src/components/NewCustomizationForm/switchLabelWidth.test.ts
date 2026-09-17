// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * KUI gives `labelPosition="left"` a fixed 160px label column
 * (`--nv-form-field-label-group-width` on `.nv-form-field-root`) and renders the label
 * with `white-space: nowrap; overflow: hidden; text-overflow: ellipsis`. A label wider
 * than the column is therefore clipped at every viewport size, and resizing the window
 * never reveals it. KUI documents the variant for "short, consistent labels" for exactly
 * this reason.
 *
 * Measured in Chromium against the generated stylesheet, the label has ~136px of usable
 * width once the info icon, its gap and the label's right padding are taken out. At the
 * form's 12px bold face that is about 22 characters: "Activation Checkpoints" (22) needs
 * 134px and fits, while "Drop Truncated Rollouts" (23) needs 139px and is clipped.
 *
 * Character count is a proxy for a proportional font, so this is a guard against the
 * obvious regression rather than a precise bound. A label that trips it is not
 * necessarily clipped — measure it before lengthening the budget.
 */
const MAX_LEFT_LABEL_CHARS = 22;

const FORM_DIR = dirname(fileURLToPath(import.meta.url));

const leftLabels = (): { file: string; label: string }[] =>
  readdirSync(FORM_DIR)
    .filter((f) => f.endsWith('.tsx') && !f.includes('.test.'))
    .flatMap((file) =>
      readFileSync(join(FORM_DIR, file), 'utf8')
        .split(/(?=<Controlled)/)
        .filter((chunk) => chunk.includes("labelPosition: 'left'"))
        .flatMap((chunk) => {
          const label = chunk.match(/slotLabel:\s*'((?:[^'\\]|\\.)*)'/)?.[1];
          return label ? [{ file, label }] : [];
        })
    );

describe('left-column switch labels', () => {
  it('finds the labels it is meant to be guarding', () => {
    expect(leftLabels().length).toBeGreaterThan(10);
  });

  it('stay inside the fixed label column', () => {
    const tooLong = leftLabels()
      .filter(({ label }) => label.length > MAX_LEFT_LABEL_CHARS)
      .map(({ file, label }) => `${label} (${label.length} chars) in ${file}`);

    expect(tooLong).toEqual([]);
  });
});
