// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  buildAgentEvaluationSpec,
  configFitIssues,
  templateFields,
} from '@studio/components/evaluation/AgentEvaluationWizard/agentEvalConfig';
import { buildEvaluationSpec } from '@studio/components/evaluation/ModelEvaluationWizard/buildEvaluationSpec';
import {
  EVALUATION_FORM_DEFAULTS,
  EMPTY_FIELD_MAPPING,
} from '@studio/components/evaluation/ModelEvaluationWizard/types';
import { toDatasetBindings } from '@studio/components/evaluation/ModelEvaluationWizard/useDatasetBindings';
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
const modelConfig = buildAgentEvaluationSpec(
  {
    ...EVALUATION_FORM_DEFAULTS,
    dataset: 'ws/chats#chats.jsonl',
    fieldMapping: {
      ...EMPTY_FIELD_MAPPING,
      messages: 'messages',
      input: 'messages[role=user].content',
      reference: 'messages[role=assistant].content',
    },
    body: { ...EVALUATION_FORM_DEFAULTS.body, judgeModel: 'ws/judge' },
  },
  'ws',
  null
);

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

describe('buildAgentEvaluationSpec', () => {
  const values = {
    ...EVALUATION_FORM_DEFAULTS,
    dataset: 'ws/data#rows.jsonl',
    fieldMapping: { ...EMPTY_FIELD_MAPPING, input: 'question', reference: 'answer' },
    body: { ...EVALUATION_FORM_DEFAULTS.body, judgeModel: 'ws/judge' },
  };

  it('writes the model wizard config unchanged when the prompt is not edited', () => {
    expect(buildAgentEvaluationSpec(values, 'ws', null)).toEqual(
      buildEvaluationSpec(values, toDatasetBindings(values.fieldMapping), 'ws')
    );
  });

  it('sends an edited prompt as the user message', () => {
    expect(buildAgentEvaluationSpec(values, 'ws', '{{ item.question }}!').prompt_template).toEqual({
      messages: [{ role: 'user', content: '{{ item.question }}!' }],
    });
  });
});
