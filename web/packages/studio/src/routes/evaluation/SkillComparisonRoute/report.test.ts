// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { makeSkillComparisonReport } from '@studio/mocks/evaluation/skillComparison';
import {
  comparisonMetrics,
  comparisonWarnings,
  MAX_REPORT_BYTES,
  parseReport,
  readReport,
  tokenUsage,
} from '@studio/routes/evaluation/SkillComparisonRoute/report';

describe('skill comparison reports', () => {
  it('accepts scorer output, preserving zero and leaving absent activation unmeasured', () => {
    const report = parseReport(JSON.stringify(makeSkillComparisonReport()));
    const activation = comparisonMetrics(report).find(
      (row) => row.name === 'skill_activation.activated'
    );
    expect(activation?.baseline).toMatchObject({ value: null, status: 'Unmeasured' });
    expect(activation?.treatment).toMatchObject({ value: 1, status: 'Measured' });
    expect(activation?.delta).toBeNull();
    expect(
      comparisonMetrics(report).find((row) => row.name === 'llm-judge.assertion_1')
    ).toMatchObject({
      baseline: { value: 0 },
      treatment: { value: 1 },
      delta: 1,
    });
  });

  it('does not infer negative activation from an absent observation', () => {
    const report = makeSkillComparisonReport();
    delete report.arms['with-skill'].activation;
    expect(parseReport(JSON.stringify(report)).arms['with-skill'].activation).toBeUndefined();
  });

  it('keeps judge labels as categorical evidence without inventing a numeric mean', () => {
    const report = makeSkillComparisonReport();
    const replay = report.arms.baseline.replay!;
    replay.scores[1].outputs.push({ name: 'assertion_1.label', value: 'met' });
    replay.summary.scores.scores.push({
      name: 'llm-judge.assertion_1.label',
      mean: null,
      count: 0,
      nan_count: 1,
    });
    replay.summary.metric_coverage['llm-judge'] = {
      'assertion_1.label': { total: 1, scored: 1, failed: 0, missing: 0 },
    };
    expect(
      comparisonMetrics(parseReport(JSON.stringify(report))).find(
        (row) => row.name === 'llm-judge.assertion_1.label'
      )
    ).toMatchObject({
      baseline: { label: 'met', value: null, status: 'Measured' },
      delta: null,
    });
  });

  it('rejects overflow numbers even inside otherwise opaque evidence', () => {
    const report = JSON.stringify(makeSkillComparisonReport()).replace(
      '"Fixture assertion evidence."',
      '1e999'
    );
    expect(() => parseReport(report)).toThrow('valid JSON');
  });

  it('preserves partial reports and withholds deltas on image drift or failed scoring', () => {
    const report = makeSkillComparisonReport();
    report.arms['with-skill'].agent_image_ref = `other/task@sha256:${'e'.repeat(64)}`;
    expect(comparisonWarnings(report)).toEqual([expect.stringContaining('different images')]);
    expect(comparisonMetrics(report).every((row) => row.delta === null)).toBe(true);
    report.arms['with-skill'].agent_image_ref = report.arms.baseline.agent_image_ref;
    report.arms['with-skill'].replay_error = 'Judge unavailable';
    expect(comparisonMetrics(report).every((row) => row.delta === null)).toBe(true);
    delete report.arms.baseline.replay;
    expect(parseReport(JSON.stringify(report)).arms.baseline.replay).toBeUndefined();
  });

  it('does not present partial metric coverage as a complete comparison', () => {
    const report = makeSkillComparisonReport();
    const replay = report.arms.baseline.replay!;
    replay.summary.metric_coverage.harbor_reward.reward = {
      total: 2,
      scored: 1,
      failed: 1,
      missing: 0,
    };
    expect(
      comparisonMetrics(report).find((row) => row.name === 'harbor_reward.reward')
    ).toMatchObject({
      baseline: { value: 1, status: 'Partial' },
      delta: null,
    });
  });

  it('counts cached input once and supports zero token usage', () => {
    const arm = makeSkillComparisonReport().arms.baseline;
    expect(tokenUsage(arm)).toEqual({
      prompt: 1000,
      cached: 600,
      completion: 100,
      total: 1100,
      uncached: 500,
    });
    arm.measurements = { prompt_tokens: 0, cache_read_tokens: 0, completion_tokens: 0 };
    expect(tokenUsage(arm)).toEqual({ prompt: 0, cached: 0, completion: 0, total: 0, uncached: 0 });
  });

  it('supports older summaries without fabricating inclusive token counts', () => {
    const arm = makeSkillComparisonReport().arms.baseline;
    delete arm.measurements;
    expect(tokenUsage(arm)).toEqual({
      prompt: null,
      cached: null,
      completion: null,
      total: null,
      uncached: 500,
    });
  });

  it('does not subtract impossible cached token counts', () => {
    const arm = makeSkillComparisonReport().arms.baseline;
    arm.measurements = { prompt_tokens: 5, cache_read_tokens: 6, completion_tokens: 1 };
    delete arm.replay;
    expect(tokenUsage(arm)).toMatchObject({ cached: null, uncached: null, total: 6 });
  });

  it('rejects malformed reports, duplicate metric names, and unpinned images', () => {
    expect(() => parseReport('<html>report</html>')).toThrow('valid JSON');
    const report = makeSkillComparisonReport();
    report.arms.baseline.agent_image_ref = 'image:latest';
    expect(() => parseReport(JSON.stringify(report))).toThrow('agent_image_ref');
    report.arms.baseline.agent_image_ref = report.arms['with-skill'].agent_image_ref;
    report.arms.baseline.replay!.summary.scores.scores.push(
      report.arms.baseline.replay!.summary.scores.scores[0]
    );
    expect(() => parseReport(JSON.stringify(report))).toThrow('unique');
  });

  it('rejects large files before reading them', async () => {
    const file = new File([''], 'large.json');
    Object.defineProperty(file, 'size', { value: MAX_REPORT_BYTES + 1 });
    const read = vi.spyOn(file, 'text');
    await expect(readReport(file)).rejects.toThrow('10 MiB');
    expect(read).not.toHaveBeenCalled();
  });
});
