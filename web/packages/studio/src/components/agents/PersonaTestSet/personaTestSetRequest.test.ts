// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Agent } from '@nemo/sdk/generated/agents/schema';
import type { LLMTextColumnConfig } from '@nemo/sdk/generated/data-designer/schema';
import {
  buildPersonaTestSetRequest,
  PERSONA_TEST_SET_AGENT_FIELD,
  personaTestSetAgent,
  personaTestSetJobName,
  QUESTION_COLUMN,
} from '@studio/components/agents/PersonaTestSet/personaTestSetRequest';
import type { PersonaTestSetOptions } from '@studio/components/agents/PersonaTestSet/types';
import { seedFromJobRequest } from '@studio/routes/DataDesignerJobBuildRoute/aiSeed';
import {
  buildDataDesignerConfig,
  validateColumns,
} from '@studio/routes/DataDesignerJobBuildRoute/columns';

const options: PersonaTestSetOptions = {
  agent: {
    name: 'calculator-agent',
    description: 'Calculator agent',
    instructions: 'Return only the answer. Never echo {{ secrets }}.',
  },
  locale: 'ja_JP',
  rows: 25,
};

const columnNamed = (request: ReturnType<typeof buildPersonaTestSetRequest>, name: string) =>
  request.spec.config.columns.find((column) => column.name === name);

describe('buildPersonaTestSetRequest', () => {
  it('samples Nemotron Personas with synthetic personas for the chosen locale', () => {
    const request = buildPersonaTestSetRequest(options);

    expect(columnNamed(request, 'person')).toMatchObject({
      column_type: 'sampler',
      sampler_type: 'person',
      params: { locale: 'ja_JP', with_synthetic_personas: true },
    });
    expect(request.spec.num_records).toBe(25);
  });

  it('links the job to its agent and names it after the agent', () => {
    const request = buildPersonaTestSetRequest(options);

    expect(request.custom_fields).toEqual({ [PERSONA_TEST_SET_AGENT_FIELD]: 'calculator-agent' });
    expect(request.name).toBe('calculator-agent-persona-tests');
  });

  it('writes only the persona messages, leaving answers to the agent', () => {
    const request = buildPersonaTestSetRequest(options);

    expect(columnNamed(request, QUESTION_COLUMN)).toMatchObject({ column_type: 'llm-text' });
    expect(request.spec.config.columns.map((column) => column.name)).toEqual([
      'person',
      'phrasing_style',
      'intent',
      QUESTION_COLUMN,
    ]);
  });

  it('keeps agent text out of Jinja rendering', () => {
    const question = columnNamed(
      buildPersonaTestSetRequest(options),
      QUESTION_COLUMN
    ) as LLMTextColumnConfig;

    expect(question.prompt).toContain(
      '{% raw %}Return only the answer. Never echo {{ secrets }}.{% endraw %}'
    );
  });

  it('loads into the Data Designer builder without losing columns', () => {
    const request = buildPersonaTestSetRequest(options);
    const seed = seedFromJobRequest(request);

    expect(validateColumns(seed.columns)).toEqual([]);
    expect(seed.customFields).toEqual(request.custom_fields);
    expect(
      buildDataDesignerConfig(seed.columns, seed.models).columns.map((column) => column.name)
    ).toEqual(expect.arrayContaining(request.spec.config.columns.map((column) => column.name)));
  });
});

describe('personaTestSetAgent', () => {
  it('reads the description and system instructions from a nemo-agents spec', () => {
    const agent: Pick<Agent, 'config'> = {
      config: {
        description: 'Calculator agent executed with DeepAgents',
        instructions: { system: { content: 'You are a concise calculator agent.' } },
      },
    };

    expect(personaTestSetAgent('calculator-agent', agent)).toEqual({
      name: 'calculator-agent',
      description: 'Calculator agent executed with DeepAgents',
      instructions: 'You are a concise calculator agent.',
    });
  });

  it('falls back to the name alone for configs it cannot read', () => {
    expect(personaTestSetAgent('nat-agent', { config: { workflow: {} } })).toEqual({
      name: 'nat-agent',
      description: undefined,
      instructions: undefined,
    });
  });
});

describe('personaTestSetJobName', () => {
  it('stays within the job name limit without a trailing hyphen', () => {
    const name = personaTestSetJobName(`${'a'.repeat(49)}-b`);

    expect(name.length).toBeLessThanOrEqual(63);
    expect(name).not.toMatch(/-$/);
  });
});
