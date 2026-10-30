// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Agent } from '@nemo/sdk/generated/agents/schema';
import type {
  CreateJobRequest,
  DataDesignerConfig,
  ExpressionColumnConfig,
  LLMTextColumnConfig,
  SamplerColumnConfig,
} from '@nemo/sdk/generated/data-designer/schema';
import type {
  PersonaAnswerStyle,
  PersonaTestSetAgent,
  PersonaTestSetOptions,
} from '@studio/components/agents/PersonaTestSet/types';
import { asRecord } from '@studio/util/guards';

/** Recorded on the Data Designer job so its details page can hand the output back to the agent. */
export const PERSONA_TEST_SET_AGENT_FIELD = 'persona_test_set_agent';

/** The model alias every generated column uses; the builder resolves it to a workspace model. */
export const PERSONA_MODEL_ALIAS = 'persona-writer';

/** Column names the agent evaluation reads, matching the sample calculator eval dataset. */
export const QUESTION_COLUMN = 'question';
export const ANSWER_COLUMN = 'answer';

export const DEFAULT_INTENTS = [
  'typical request',
  'multi-step request',
  'edge case',
  'request outside what the agent is for',
];

export const PHRASING_STYLES = [
  'terse',
  'conversational',
  'rambling, with extra personal context',
  'informal, with typos',
  'indirect or ambiguous',
];

const ARITHMETIC_OPERATIONS = ['addition', 'subtraction', 'multiplication', 'division'];

const MAX_JOB_NAME_LENGTH = 63;

/** Agent text is data, not template: Data Designer would otherwise render any `{{ }}` in it. */
const literal = (text: string): string =>
  `{% raw %}${text.replaceAll('{% endraw %}', '')}{% endraw %}`;

const describeAgent = ({ name, description, instructions }: PersonaTestSetAgent): string =>
  [
    `Name: ${literal(name)}`,
    description?.trim() ? `Description: ${literal(description.trim())}` : null,
    instructions?.trim() ? `Instructions it follows:\n${literal(instructions.trim())}` : null,
  ]
    .filter(Boolean)
    .join('\n');

const PERSONA_PROFILE = `About you: {{ person.persona }}
Age: {{ person.age }}. Occupation: {{ person.occupation }}. Education: {{ person.education_level }}.`;

const sampler = (
  name: string,
  sampler_type: SamplerColumnConfig['sampler_type'],
  params: SamplerColumnConfig['params'],
  convert_to?: string
): SamplerColumnConfig => ({
  name,
  column_type: 'sampler',
  sampler_type,
  params,
  ...(convert_to ? { convert_to } : {}),
});

const llmText = (name: string, prompt: string): LLMTextColumnConfig => ({
  name,
  column_type: 'llm-text',
  prompt,
  model_alias: PERSONA_MODEL_ALIAS,
  extract_reasoning_content: true,
});

const expression = (name: string, expr: string): ExpressionColumnConfig => ({
  name,
  column_type: 'expression',
  expr,
});

const questionPrompt = (agent: PersonaTestSetAgent, task: string): string =>
  `You are role-playing a real person who uses an AI agent.

${PERSONA_PROFILE}

The agent you are writing to:
${describeAgent(agent)}

Write one message you would send this agent. ${task} Phrase it in a {{ phrasing_style }} way.
Let your background shape what you ask about and how you say it. Return only the message.`;

const rubricColumns = (agent: PersonaTestSetAgent) => [
  sampler('intent', 'category', { values: DEFAULT_INTENTS }),
  llmText(QUESTION_COLUMN, questionPrompt(agent, 'It should be a {{ intent }}.')),
  llmText(
    ANSWER_COLUMN,
    `An AI agent received the message below. Write the grading criteria for its reply.

The agent:
${describeAgent(agent)}

Message: {{ ${QUESTION_COLUMN} }}

In one or two sentences starting with "Answer must", state what a correct reply contains. If the
request is outside what the agent is for, state that the reply must say so instead of answering.
Return only the criteria.`
  ),
];

/** Jinja's `/` is true division; results are rounded so the rubric quotes a stable number. */
const ARITHMETIC_EXPECTED = `{%- if operation == 'addition' -%}{{ a + b }}
{%- elif operation == 'subtraction' -%}{{ a - b }}
{%- elif operation == 'multiplication' -%}{{ a * b }}
{%- else -%}{{ (a / b) | round(2) }}{%- endif -%}`;

const arithmeticColumns = (agent: PersonaTestSetAgent) => [
  sampler('operation', 'category', { values: ARITHMETIC_OPERATIONS }),
  sampler('a', 'uniform', { low: 1, high: 1000, decimal_places: 0 }, 'int'),
  sampler('b', 'uniform', { low: 1, high: 100, decimal_places: 0 }, 'int'),
  expression('expected', ARITHMETIC_EXPECTED),
  llmText(
    QUESTION_COLUMN,
    questionPrompt(
      agent,
      'It must ask for the {{ operation }} of {{ a }} and {{ b }}, set in a situation from your life, using those exact numbers.'
    )
  ),
  expression(ANSWER_COLUMN, 'Answer must contain {{ expected }}'),
];

const COLUMNS_BY_STYLE: Record<
  PersonaAnswerStyle,
  (agent: PersonaTestSetAgent) => DataDesignerConfig['columns']
> = {
  rubric: rubricColumns,
  arithmetic: arithmeticColumns,
};

/** Reads what personas need to know from a `nemo-agents-spec-v1` config; other formats give none. */
export const personaTestSetAgent = (
  name: string,
  agent?: Pick<Agent, 'description' | 'config'>
): PersonaTestSetAgent => {
  const config = asRecord(agent?.config);
  const system = asRecord(asRecord(config?.instructions)?.system)?.content;
  const description = agent?.description ?? config?.description;
  return {
    name,
    description: typeof description === 'string' ? description : undefined,
    instructions: typeof system === 'string' ? system : undefined,
  };
};

export const personaTestSetJobName = (agentName: string): string =>
  `${agentName}-persona-tests`.slice(0, MAX_JOB_NAME_LENGTH).replace(/-+$/, '');

/**
 * A Data Designer job that samples Nemotron Personas and has each persona write a message to the
 * agent, with the expected answer alongside. Always the `person` sampler: test sets are meant to
 * carry census-grounded diversity, which Faker people do not.
 */
export const buildPersonaTestSetRequest = ({
  agent,
  locale,
  rows,
  answerStyle,
}: PersonaTestSetOptions): CreateJobRequest => ({
  name: personaTestSetJobName(agent.name),
  description: `Nemotron persona test set for agent ${agent.name}`,
  custom_fields: { [PERSONA_TEST_SET_AGENT_FIELD]: agent.name },
  spec: {
    num_records: rows,
    config: {
      model_configs: [{ alias: PERSONA_MODEL_ALIAS, model: '', provider: '' }],
      columns: [
        sampler('person', 'person', { locale, with_synthetic_personas: true }),
        sampler('phrasing_style', 'category', { values: PHRASING_STYLES }),
        ...COLUMNS_BY_STYLE[answerStyle](agent),
      ],
    },
  },
});
