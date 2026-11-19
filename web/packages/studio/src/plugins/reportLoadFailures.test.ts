// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { REPORT_LIMITS, clampIssues } from '@studio/plugins/reportLoadFailures';
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

  it('leaves in-limit issues unchanged', () => {
    const small: PluginLoadIssue = {
      code: 'PLUGIN_LOAD_FAILED',
      subject: '',
      message: 'boom',
      steps: ['one', 'two'],
    };

    expect(clampIssues([small])).toEqual([small]);
  });

  describe('REPORT_LIMITS', () => {
    const limitIssue = (extra = 0): PluginLoadIssue => ({
      code: 'PLUGIN_LOAD_FAILED',
      subject: 's'.repeat(REPORT_LIMITS.subject + extra),
      message: 'm'.repeat(REPORT_LIMITS.message + extra),
      steps: Array.from({ length: REPORT_LIMITS.steps + extra }, () =>
        't'.repeat(REPORT_LIMITS.step + extra)
      ),
    });

    it('are the documented backend limits', () => {
      expect(REPORT_LIMITS).toEqual({
        issues: 10,
        subject: 200,
        message: 500,
        steps: 8,
        step: 300,
      });
    });

    it('leaves an input exactly at every limit unchanged', () => {
      const input = Array.from({ length: REPORT_LIMITS.issues }, () => limitIssue());

      expect(clampIssues(input)).toEqual(input);
    });

    it('cuts an input one over every limit back to the limits', () => {
      const input = Array.from({ length: REPORT_LIMITS.issues + 1 }, () => limitIssue(1));

      const clamped = clampIssues(input);

      expect(clamped).toEqual(Array.from({ length: REPORT_LIMITS.issues }, () => limitIssue()));
    });
  });
});
