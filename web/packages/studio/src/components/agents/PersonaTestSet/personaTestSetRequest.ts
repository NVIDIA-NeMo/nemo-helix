// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Agent } from '@nemo/sdk/generated/agents/schema';
import type {
  CreateJobRequest,
  LLMTextColumnConfig,
  SamplerColumnConfig,
} from '@nemo/sdk/generated/data-designer/schema';
import type {
  PersonaTestSetAgent,
  PersonaTestSetOptions,
} from '@studio/components/agents/PersonaTestSet/types';
import { asRecord } from '@studio/util/guards';

/** Recorded on the Data Designer job so its details page can hand the output back to the agent. */
export const PERSONA_TEST_SET_AGENT_FIELD = 'persona_test_set_agent';

/** The model alias every generated column uses; the builder resolves it to a workspace model. */
export const PERSONA_MODEL_ALIAS = 'persona-writer';

/** The column the agent evaluation sends to the agent, matching the sample calculator eval dataset. */
export const QUESTION_COLUMN = 'question';

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
  params: SamplerColumnConfig['params']
): SamplerColumnConfig => ({ name, column_type: 'sampler', sampler_type, params });

const llmText = (name: string, prompt: string): LLMTextColumnConfig => ({
  name,
  column_type: 'llm-text',
  prompt,
  model_alias: PERSONA_MODEL_ALIAS,
  extract_reasoning_content: true,
});

const questionPrompt = (agent: PersonaTestSetAgent): string =>
  `You are role-playing a real person who uses an AI agent.

${PERSONA_PROFILE}

The agent you are writing to:
${describeAgent(agent)}

Write one message you would send this agent. It should be a {{ intent }}. Phrase it in a
{{ phrasing_style }} way. Let your background shape what you ask about and how you say it.
Return only the message.`;

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
 * agent. No answer is generated: the agent answers each message itself during evaluation. Always
 * the `person` sampler: test sets are meant to carry census-grounded diversity, which Faker
 * people do not.
 */
export const buildPersonaTestSetRequest = ({
  agent,
  locale,
  rows,
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
        sampler('intent', 'category', { values: DEFAULT_INTENTS }),
        llmText(QUESTION_COLUMN, questionPrompt(agent)),
      ],
    },
  },
});
