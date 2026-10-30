// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  renderPromptTemplate,
  UnsupportedTemplateError,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/promptTemplate';

// Expected outputs below were produced by the evaluator's own `render_template` (Python Jinja2).
describe('renderPromptTemplate', () => {
  it('renders the email-security-triage template exactly as Jinja does', () => {
    const template = [
      '{{ item.user_message }}',
      '{% for e in item.emails %}',
      '--- Message {{ loop.index }} ---',
      'Subject: {{ e.subject }}',
      'From: {{ e.from }}',
      '',
      '{{ e.body }}',
      '{% endfor %}',
    ].join('\n');
    const row = {
      user_message: 'is this legit?',
      emails: [
        { from: 'a@x.com', subject: 'S1', body: 'B1' },
        { from: 'b@y.com', subject: 'S2', body: 'B2' },
      ],
    };

    expect(renderPromptTemplate(template, row)).toBe(
      'is this legit?\n\n--- Message 1 ---\nSubject: S1\nFrom: a@x.com\n\nB1\n\n' +
        '--- Message 2 ---\nSubject: S2\nFrom: b@y.com\n\nB2\n'
    );
  });

  it('honors whitespace control, if/else, loop.last and the trailing newline rule', () => {
    const template =
      '{%- for e in item.xs -%}\n  [{{ e }}{% if loop.last %}.{% else %},{% endif %}]\n{%- endfor %}\n';

    expect(renderPromptTemplate(template, { xs: ['a', 'b'] })).toBe('[a,][b.]');
  });

  it('renders scalars as Python would, and the length filter', () => {
    expect(
      renderPromptTemplate('{{ item.emails | length }} {{ n }} {{ flag }} {{ nothing }}\n', {
        emails: [1, 2, 3],
        n: 2,
        flag: true,
        nothing: null,
      })
    ).toBe('3 2 True None');
  });

  it('renders a Parquet INT64 BigInt without losing precision', () => {
    expect(renderPromptTemplate('{{ id }}', { id: 9007199254740993n })).toBe('9007199254740993');
  });

  it('ignores comments', () => {
    expect(renderPromptTemplate('{# note #}{{ q }}', { q: 'hi' })).toBe('hi');
  });

  it.each([
    ['a missing field', '{{ item.missing }}'],
    ['an inherited property', '{% if item.toString %}x{% endif %}'],
    ['an unsupported filter', '{{ item.q | upper }}'],
    ['an unsupported tag', '{% set x = 1 %}{{ x }}'],
    ['an unclosed loop', '{% for e in item.xs %}{{ e }}'],
  ])('rejects %s', (_, template) => {
    expect(() => renderPromptTemplate(template, { q: 'hi', xs: [] })).toThrow(
      UnsupportedTemplateError
    );
  });
});
