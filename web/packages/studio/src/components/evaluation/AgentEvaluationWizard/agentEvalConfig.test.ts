// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  buildAgentEvaluationSpec,
  configFitIssues,
  defaultAgentPrompt,
  defaultJudgePrompt,
  templateFields,
} from '@studio/components/evaluation/AgentEvaluationWizard/agentEvalConfig';
import { EVALUATION_FORM_DEFAULTS } from '@studio/components/evaluation/ModelEvaluationWizard/types';
import {
  type DatasetEvalSpec,
  parseUploadedDatasetConfig,
} from '@studio/components/evaluation/submitEvaluationJob';
import { readFileSync } from 'fs';
import { join } from 'path';

const EXAMPLE_DIR = join(
  __dirname,
  '../../../../../../../plugins/nemo-agents/examples/nemo-agent-config/email-security-triage'
);
const triageConfig = parseUploadedDatasetConfig(
  readFileSync(join(EXAMPLE_DIR, 'eval-config.dataset-driven.yml'), 'utf-8')
);
const triageColumns = Object.keys(
  JSON.parse(readFileSync(join(EXAMPLE_DIR, 'dataset.jsonl'), 'utf-8').split('\n')[0])
);

/** What the model wizard saves for an OpenAI messages dataset: input and reference bound by
 *  role into the ``messages`` column. */
const modelConfig: DatasetEvalSpec = {
  dataset: 'ws/chats#chats.jsonl',
  prompt_template: { messages: [{ role: 'user', content: '{{input}}' }] },
  field_mapping: {
    input: 'messages[role=user].content',
    reference: 'messages[role=assistant].content',
  },
  metrics: [],
};

const withPrompt = (prompt_template: DatasetEvalSpec['prompt_template']): DatasetEvalSpec => ({
  dataset: '',
  metrics: [],
  prompt_template,
});

describe('configFitIssues', () => {
  it('passes the email-triage example against its own dataset', () => {
    expect(configFitIssues(triageConfig, triageColumns)).toEqual([]);
  });

  it('flags a messages-mapped model config against a dataset with no messages column', () => {
    const issues = configFitIssues(modelConfig, triageColumns);
    expect(issues).toContain(
      'Maps "input" to "messages[role=user].content", but the dataset has no "messages" column.'
    );
    expect(issues.every((issue) => !issue.startsWith('Reads'))).toBe(true);
  });

  it('passes the same model config against a messages dataset', () => {
    expect(configFitIssues(modelConfig, ['messages'])).toEqual([]);
  });

  it('flags a field the dataset does not have, in either item syntax', () => {
    expect(configFitIssues(withPrompt('{{ item.missing }}'), ['present'])).toEqual([
      'Reads "missing", which is not a dataset column or a mapped field.',
    ]);
    expect(configFitIssues(withPrompt("{{ item['present'] }}"), ['present'])).toEqual([]);
  });

  it('flags a dict prompt that renders to neither messages nor a prompt', () => {
    expect(configFitIssues(withPrompt({ text: '{{ item.q }}' }), ['q'])).toEqual([
      'The prompt template must be text or a chat "messages" list.',
    ]);
  });
});

describe('templateFields', () => {
  it('ignores loop locals, filters, tests, literals and the response', () => {
    expect(
      templateFields(
        '{% for e in item.emails %}{{ e.body | trim }}{{ loop.index }}{% endfor %}' +
          "{{ 'x' if sample.output_text is string else name }}{# {{ item.hidden }} #}"
      )
    ).toEqual(['emails', 'name']);
  });
});

describe('default prompts', () => {
  it('render every column under its name, and the judge also reads the response', () => {
    expect(defaultAgentPrompt(['user_message', 'emails'])).toBe(
      'user_message:\n{{ item.user_message }}\n\nemails:\n{{ item.emails }}'
    );
    expect(defaultJudgePrompt(['expected'])).toBe(
      'expected:\n{{ item.expected }}\n\nResponse to evaluate:\n{{ sample.output_text }}'
    );
  });

  it('index a column whose name is not an identifier', () => {
    expect(defaultAgentPrompt(['my field'])).toBe("my field:\n{{ item['my field'] }}");
  });

  it('fit the dataset they were generated from', () => {
    for (const prompt of [defaultAgentPrompt, defaultJudgePrompt]) {
      expect(configFitIssues(withPrompt(prompt(triageColumns)), triageColumns)).toEqual([]);
    }
  });
});

describe('buildAgentEvaluationSpec', () => {
  const judge = {
    judgeModel: 'ws/judge',
    judgePrompt: defaultJudgePrompt(['answer']),
    scores: EVALUATION_FORM_DEFAULTS.body.scores,
  };

  it('writes a text prompt, no field mapping and one LLM judge with the given prompt', () => {
    const spec = buildAgentEvaluationSpec('ws/data#rows.jsonl', '{{ item.question }}', judge, 'ws');

    expect(spec).toEqual({
      dataset: 'ws/data#rows.jsonl',
      prompt_template: '{{ item.question }}',
      metrics: [expect.objectContaining({ metric_type: 'llm-judge' })],
    });
    expect(spec.metrics[0].payload.metric).toMatchObject({
      model: { name: 'judge' },
      prompt_template: { messages: [{ role: 'user', content: judge.judgePrompt }] },
    });
  });
});
