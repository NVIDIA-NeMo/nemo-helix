// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  classifyJudgePromptVariables,
  expressionRoot,
} from '@studio/routes/evaluation/EvaluationNewRoute/judgePromptVariables';

const COLUMNS = ['query', 'answer', 'difficulty'];
const MAPPING = { input: 'query', reference: 'answer' } as const;

const classify = (template: string) => classifyJudgePromptVariables(template, COLUMNS, MAPPING);

describe('expressionRoot', () => {
  it.each([
    ['input', 'input'],
    ['  input  ', 'input'],
    ['messages[1].content', 'messages'],
    ['sample.output_text', 'sample'],
    ['input | upper', 'input'],
    ['item.query', 'item'],
  ])('reads %s as %s', (expression, root) => {
    expect(expressionRoot(expression)).toBe(root);
  });

  it.each(["'literal'", '1 + 2', ''])('declines to guess at %s', (expression) => {
    expect(expressionRoot(expression)).toBeNull();
  });
});

describe('classifyJudgePromptVariables', () => {
  it('accepts canonical names, wrappers and the candidate payload', () => {
    const report = classify(
      'Input: {{input}}\nGround Truth: {{reference}}\n{{ context }}\n' +
        '{{ messages[0].content }}\n{{sample.output_text}}\n{{item.query}}\n{{scores}}'
    );
    expect(report).toEqual({ invalid: [], nonPortable: [], malformed: false });
  });

  it('accepts a filtered canonical name', () => {
    expect(classify('{{ input | upper }}').invalid).toEqual([]);
  });

  it('accepts any index into the messages array', () => {
    expect(classify('{{ messages[9].content }}').invalid).toEqual([]);
  });

  it('reports a mapped column as non-portable, naming its canonical field', () => {
    const report = classify('Question: {{query}}');
    expect(report.invalid).toEqual([]);
    expect(report.nonPortable).toEqual([{ token: 'query', canonical: 'input' }]);
  });

  it('leaves an unmapped column alone, having no portable form to suggest', () => {
    expect(classify('Difficulty: {{difficulty}}')).toEqual({
      invalid: [],
      nonPortable: [],
      malformed: false,
    });
  });

  it('reports a name that resolves to nothing', () => {
    expect(classify('{{feedback}}').invalid).toEqual(['feedback']);
  });

  it('ignores literals and statement blocks', () => {
    expect(classify("{{ 'verbatim' }} {% if input %}x{% endif %}").invalid).toEqual([]);
  });

  it('reports a variable inserted inside another variable', () => {
    expect(classify('Input: {{{{reference}}input}}').malformed).toBe(true);
  });

  it('reports a lone closing brace pair', () => {
    expect(classify('Input: {{input}} }}').malformed).toBe(true);
  });

  it('reports a lone opening brace pair', () => {
    expect(classify('Input: {{ input').malformed).toBe(true);
  });

  it('does not cry malformed over a well-formed prompt', () => {
    expect(classify('{{input}} {{ messages[1].content }} {{sample.output_text}}').malformed).toBe(
      false
    );
  });

  it('reports each bad root once', () => {
    expect(classify('{{nope}} {{nope}} {{ nope }}').invalid).toEqual(['nope']);
  });
});
