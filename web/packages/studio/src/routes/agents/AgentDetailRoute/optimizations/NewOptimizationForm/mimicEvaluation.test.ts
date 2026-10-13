// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  DatasetEvalSpec,
  InlineMetricBundle,
} from '@studio/components/evaluation/submitEvaluationJob';
import {
  CANDIDATE_MARKER,
  mimicEvaluation,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/mimicEvaluation';

const judge: InlineMetricBundle = {
  bundle_kind: 'metric-bundle',
  bundle_format_version: 'v1',
  metric_type: 'llm-judge',
  payload: {
    kind: 'inline',
    metric: {
      type: 'llm-judge',
      model: 'default/original-judge',
      scores: [{ name: 'accuracy', minimum: 0, maximum: 1 }],
      inference: { max_tokens: 1024, extra_body: { nvext: { max_thinking_tokens: 256 } } },
      prompt_template: {
        messages: [
          {
            role: 'user',
            content:
              'Score {{ item.user_message }} over {{ item.emails | length }} messages. Expected verdicts: {{ item.expected_answer }}. Agent response: {{ sample.output_text }}. Compare verdicts in order, ignoring reasoning.',
          },
        ],
      },
    },
  },
};
const routing: InlineMetricBundle = {
  bundle_kind: 'metric-bundle',
  bundle_format_version: 'v1',
  metric_type: 'string-check',
  payload: { kind: 'inline', metric: { type: 'string-check' } },
};
const rows = [
  {
    user_message: 'is this legit?',
    emails: [{ subject: 'Quota', body: 'Click here' }],
    expected_answer: 'phishing',
    expected_tool: 'triage_message',
  },
  {
    user_message: '',
    emails: [
      { subject: 'Receipt', body: 'Confirmed' },
      { subject: 'Urgent', body: 'Pay now' },
    ],
    expected_answer: 'benign, phishing',
    expected_tool: 'review_messages',
  },
];
const spec: DatasetEvalSpec = {
  dataset: rows,
  prompt_template:
    '{{ item.user_message }}{% for e in item.emails %}\n{{ loop.index }}. {{ e.subject }}\n{{ e.body }}{% endfor %}',
  metrics: [judge, routing],
};

describe('mimicEvaluation', () => {
  it('stages the original prompts and per-row accuracy rubric without a reference mapping', () => {
    const result = mimicEvaluation({ spec, records: rows });
    expect(result.rows[0].question).toBe('is this legit?\n1. Quota\nClick here');
    expect(result.rows[1].question).toBe('\n1. Receipt\nConfirmed\n2. Urgent\nPay now');
    expect(result.rows[0].answer).toContain(
      `Expected verdicts: phishing. Agent response: ${CANDIDATE_MARKER}`
    );
    expect(result.rows[1].answer).toContain(
      'over 2 messages. Expected verdicts: benign, phishing.'
    );
    expect(result.rows[0].answer).toContain('return the original "accuracy" value as "score"');
    expect(result.rows[0].question).not.toContain('Expected verdicts');
    expect(result.skippedMetrics).toEqual(['string-check']);
    expect(result.model).toBe('default/original-judge');
    expect(result.inference).toEqual(judge.payload.metric.inference);
    expect(spec.metrics[0]).toEqual(judge);
  });

  it('uses the authored mapping and fails when it does not resolve instead of falling back', () => {
    const mapped = {
      ...spec,
      field_mapping: { reference: null, custom: { expected_answer: 'gold.verdict' } },
    };
    const records = [{ ...rows[0], gold: { verdict: 'benign' } }];
    expect(mimicEvaluation({ spec: mapped, records }).rows[0].answer).toContain(
      'Expected verdicts: benign.'
    );
    expect(() => mimicEvaluation({ spec: mapped, records: rows })).toThrow(
      /undefined value "item.expected_answer"/
    );
  });

  it('keeps task references out of the agent prompt while rendering the task judge', () => {
    const taskJudge = {
      ...judge,
      payload: {
        kind: 'inline' as const,
        metric: {
          ...judge.payload.metric,
          prompt_template: '{{ item.reference.answer }} vs {{ sample.output_text }}',
        },
      },
    };
    const task = {
      id: 't',
      intent: 'Classify',
      inputs: { instruction: 'Check this mail' },
      reference: { answer: 'phishing' },
      metrics: [taskJudge],
    };
    const result = mimicEvaluation({ spec: { tasks: [task] }, records: [task] });
    expect(result.rows[0]).toMatchObject({
      id: 't',
      question: 'Check this mail',
      answer: expect.stringContaining(`phishing vs ${CANDIDATE_MARKER}`),
    });
  });

  it('rejects multiple judges, custom score parsers, and unsupported metrics instead of guessing', () => {
    expect(() =>
      mimicEvaluation({ spec: { ...spec, metrics: [judge, judge] }, records: rows })
    ).toThrow(/exactly one LLM judge/);
    expect(() => mimicEvaluation({ spec: { ...spec, metrics: [routing] }, records: rows })).toThrow(
      /exactly one LLM judge/
    );
    const custom = {
      ...judge,
      payload: {
        kind: 'inline' as const,
        metric: {
          ...judge.payload.metric,
          scores: [{ name: 'accuracy', minimum: 0, maximum: 1, parser: { type: 'regex' } }],
        },
      },
    };
    expect(() => mimicEvaluation({ spec: { ...spec, metrics: [custom] }, records: rows })).toThrow(
      /default parser/
    );
  });

  it('rejects unsupported template syntax with a useful error', () => {
    expect(() =>
      mimicEvaluation({
        spec: { ...spec, prompt_template: '{{ item.user_message | upper }}' },
        records: rows,
      })
    ).toThrow(/filter "upper"/);
  });
});
