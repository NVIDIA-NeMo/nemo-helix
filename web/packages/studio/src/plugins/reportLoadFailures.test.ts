// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { clampIssues } from '@studio/plugins/reportLoadFailures';
import type { PluginLoadIssue } from '@studio/plugins/types';

describe('clampIssues', () => {
  const longIssue = (n: number): PluginLoadIssue => ({
    code: 'PLUGIN_SPECIFIER_UNMAPPED',
    subject: `subject-${n}-${'s'.repeat(400)}`,
    message: `message-${n}-${'m'.repeat(600)}`,
    steps: Array.from({ length: 12 }, (_, k) => `step-${n}-${k}-${'t'.repeat(400)}`),
  });

  it('caps issues at 10, steps at 8, and string lengths, keeping prefixes', () => {
    const input = Array.from({ length: 12 }, (_, n) => longIssue(n));

    const clamped = clampIssues(input);

    expect(clamped).toHaveLength(10);
    clamped.forEach((out, n) => {
      expect(out.code).toBe(input[n].code);
      expect(out.subject).toHaveLength(200);
      expect(input[n].subject.startsWith(out.subject)).toBe(true);
      expect(out.message).toHaveLength(500);
      expect(input[n].message.startsWith(out.message)).toBe(true);
      expect(out.steps).toHaveLength(8);
      out.steps.forEach((step, k) => {
        expect(step).toHaveLength(300);
        expect(input[n].steps[k].startsWith(step)).toBe(true);
      });
    });
  });

  it('does not mutate its input', () => {
    const input = Array.from({ length: 12 }, (_, n) => longIssue(n));
    const snapshot = structuredClone(input);

    clampIssues(input);

    expect(input).toEqual(snapshot);
  });

  it('leaves in-limit issues unchanged', () => {
    const small: PluginLoadIssue = {
      code: 'PLUGIN_LOAD_FAILED',
      subject: '',
      message: 'boom',
      steps: ['one', 'two'],
    };

    expect(clampIssues([small])).toEqual([small]);
  });
});
