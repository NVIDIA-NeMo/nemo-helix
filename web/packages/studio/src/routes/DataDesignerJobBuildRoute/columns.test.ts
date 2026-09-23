// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SamplerType } from '@nemo/sdk/generated/data-designer/schema';
import { COLUMN_TYPE_GROUPS } from '@studio/components/AddColumnPalette/constants';
import type { ColumnTypeOption } from '@studio/components/AddColumnPalette/types';
import { FILESET_TEMPLATES } from '@studio/components/CreateFilesetStart/templates';
import {
  type BuilderColumn,
  buildColumnsFromConfig,
  buildColumnsFromTemplate,
  buildDataDesignerConfig,
  buildGraph,
  defaultColumnName,
  defaultColumnValues,
  extractJinjaReferences,
  findColumnOption,
  topologicalSortColumns,
  validateColumnName,
  validateColumns,
} from '@studio/routes/DataDesignerJobBuildRoute/columns';

const optionFor = (columnType: string, samplerType?: string): ColumnTypeOption => {
  const found = findColumnOption({ columnType, samplerType } as never);
  if (!found) throw new Error(`no palette option for ${columnType}/${samplerType}`);
  return found;
};

const column = (
  id: string,
  name: string,
  columnType: string,
  values: Record<string, string>,
  samplerType?: string
): BuilderColumn => ({ id, name, option: optionFor(columnType, samplerType), values });

describe('extractJinjaReferences', () => {
  it('pulls identifiers out of {{ }} tokens, including filters and whitespace variants', () => {
    expect(extractJinjaReferences('Write about {{ topic }} for {{audience}}')).toEqual([
      'topic',
      'audience',
    ]);
    expect(extractJinjaReferences('{{- name | upper }}')).toEqual(['name']);
    expect(extractJinjaReferences('no references here')).toEqual([]);
  });
});

describe('validateColumnName', () => {
  it('requires a non-empty valid identifier that is unique', () => {
    expect(validateColumnName('', new Set())).toMatch(/required/i);
    expect(validateColumnName('2bad', new Set())).toMatch(/letters/i);
    expect(validateColumnName('with space', new Set())).toMatch(/letters/i);
    expect(validateColumnName('topic', new Set(['topic']))).toMatch(/already exists/i);
    expect(validateColumnName('topic', new Set(['other']))).toBeNull();
  });
});

describe('defaultColumnName', () => {
  it('derives a unique name from the column type, skipping taken names', () => {
    const option = optionFor('llm-text');
    expect(defaultColumnName(option, new Set())).toBe('llm_text_1');
    expect(defaultColumnName(option, new Set(['llm_text_1']))).toBe('llm_text_2');
  });

  it('uses the sampler sub-type when present', () => {
    const option = optionFor('sampler', 'uuid');
    expect(defaultColumnName(option, new Set())).toBe('uuid_1');
  });
});

describe('defaultColumnValues', () => {
  it('turns on reasoning extraction for every LLM column type', () => {
    for (const columnType of ['llm-text', 'llm-code', 'llm-structured', 'llm-judge']) {
      expect(defaultColumnValues(optionFor(columnType))).toMatchObject({
        extract_reasoning_content: 'true',
      });
    }
  });

  it('leaves column types without field defaults empty', () => {
    expect(defaultColumnValues(optionFor('expression'))).toEqual({});
  });
});

describe('buildColumnsFromTemplate', () => {
  it('resolves specs into placed columns with sequential ids and seeded values', () => {
    const columns = buildColumnsFromTemplate([
      { columnType: 'sampler', samplerType: 'category', name: 'domain' } as never,
      {
        columnType: 'llm-text',
        name: 'instruction',
        values: { prompt: 'About {{ domain }}', model_alias: 'default' },
      },
    ]);

    expect(columns.map((c) => c.id)).toEqual(['col-0', 'col-1']);
    expect(columns.map((c) => c.name)).toEqual(['domain', 'instruction']);
    expect(columns[0].option.samplerType).toBe('category');
    expect(columns[1].values).toEqual({
      prompt: 'About {{ domain }}',
      model_alias: 'default',
      extract_reasoning_content: 'true',
    });
  });

  it('lets a template override a field default', () => {
    const [column] = buildColumnsFromTemplate([
      {
        columnType: 'llm-text',
        name: 'answer',
        values: { extract_reasoning_content: 'false' },
      },
    ]);
    expect(column.values.extract_reasoning_content).toBe('false');
  });

  it('numbers ids from startId and skips unresolvable specs', () => {
    const columns = buildColumnsFromTemplate(
      [
        { columnType: 'not-a-type', name: 'nope' } as never,
        { columnType: 'llm-text', name: 'answer' },
      ],
      5
    );

    expect(columns).toHaveLength(1);
    expect(columns[0]).toMatchObject({ id: 'col-5', name: 'answer' });
  });
});

describe('buildGraph', () => {
  it('draws an edge only when a column references another via Jinja2', () => {
    const columns = [
      column('a', 'topic', 'seed-dataset', {}),
      column('b', 'question', 'llm-text', {
        prompt: 'Ask about {{ topic }}',
        model_alias: 'default',
      }),
    ];

    const { nodes, edges } = buildGraph(columns);

    expect(nodes).toHaveLength(2);
    expect(edges).toEqual([{ source: 'a', target: 'b' }]);
    // The dependency is surfaced as a tag on the dependent node.
    expect(nodes.find((n) => n.id === 'b')?.data.tags).toEqual(['{{topic}}']);
  });

  it('leaves unreferenced columns as independent roots', () => {
    const columns = [
      column('a', 'topic', 'seed-dataset', {}),
      column('b', 'note', 'llm-text', { prompt: 'A static prompt', model_alias: 'default' }),
    ];

    expect(buildGraph(columns).edges).toEqual([]);
  });

  it('ignores references to names that are not real columns and self-references', () => {
    const columns = [
      column('a', 'summary', 'llm-text', {
        prompt: 'Summarize {{ summary }} and {{ missing }}',
        model_alias: 'default',
      }),
    ];

    expect(buildGraph(columns).edges).toEqual([]);
  });

  it('derives edges from column-name fields (embedding target, validation targets)', () => {
    const columns = [
      column('a', 'text', 'seed-dataset', {}),
      column('b', 'other', 'seed-dataset', {}),
      column('c', 'text_vec', 'embedding', { target_column: 'text', model_alias: 'default' }),
      column('d', 'checks', 'validation', {
        target_columns: 'text, other',
        validator_type: 'code',
      }),
    ];

    const { edges } = buildGraph(columns);

    expect(edges).toContainEqual({ source: 'a', target: 'c' });
    expect(edges).toContainEqual({ source: 'a', target: 'd' });
    expect(edges).toContainEqual({ source: 'b', target: 'd' });
    expect(edges).toHaveLength(3);
  });
});

describe('sampler columns', () => {
  it('nests category values under the required params object', () => {
    const columns = [
      column('a', 'domain', 'sampler', { values: 'science, history, , arts' }, 'category'),
    ];

    expect(buildDataDesignerConfig(columns).columns[0]).toEqual({
      name: 'domain',
      column_type: 'sampler',
      sampler_type: 'category',
      params: { values: ['science', 'history', 'arts'] },
    });
  });

  it('keeps convert_to at the top level alongside params', () => {
    const columns = [
      column('a', 'domain', 'sampler', { values: 'a, b', convert_to: 'str' }, 'category'),
    ];

    expect(buildDataDesignerConfig(columns).columns[0]).toMatchObject({
      params: { values: ['a', 'b'] },
      convert_to: 'str',
    });
  });

  it('requires category values', () => {
    const columns = [column('a', 'domain', 'sampler', {}, 'category')];

    expect(validateColumns(columns)).toContainEqual(expect.stringContaining('Categories'));
  });

  it('coerces numeric sampler params to numbers', () => {
    const columns = [
      column(
        'a',
        'score',
        'sampler',
        { mean: '1.5', stddev: '0.25', decimal_places: '2' },
        'gaussian'
      ),
    ];

    expect(buildDataDesignerConfig(columns).columns[0]).toEqual({
      name: 'score',
      column_type: 'sampler',
      sampler_type: 'gaussian',
      params: { mean: 1.5, stddev: 0.25, decimal_places: 2 },
    });
  });

  it('serializes boolean switches and a numeric weight list', () => {
    const uuid = column('a', 'id', 'sampler', { short_form: 'true', uppercase: 'false' }, 'uuid');
    const category = column(
      'b',
      'domain',
      'sampler',
      { values: 'a, b, c', weights: '3, 1, 1' },
      'category'
    );

    const built = buildDataDesignerConfig([uuid, category]).columns;
    expect(built[0]).toMatchObject({ params: { short_form: 'true', uppercase: 'false' } });
    expect(built[1]).toMatchObject({ params: { values: ['a', 'b', 'c'], weights: [3, 1, 1] } });
  });

  it('parses JSON sampler params (subcategory mapping)', () => {
    const columns = [
      column(
        'a',
        'sub',
        'sampler',
        { category: 'domain', values: '{ "sci": ["physics"], "art": ["music"] }' },
        'subcategory'
      ),
    ];

    expect(buildDataDesignerConfig(columns).columns[0]).toMatchObject({
      params: { category: 'domain', values: { sci: ['physics'], art: ['music'] } },
    });
  });

  it('draws edges from sampler column-name references (subcategory parent, timedelta reference)', () => {
    const columns = [
      column('a', 'domain', 'sampler', { values: 'x, y' }, 'category'),
      column(
        'b',
        'sub',
        'sampler',
        { category: 'domain', values: '{ "x": ["x1"] }' },
        'subcategory'
      ),
      column('c', 'created', 'sampler', { start: '2020-01-01', end: '2024-01-01' }, 'datetime'),
      column(
        'd',
        'shipped',
        'sampler',
        { dt_min: '1', dt_max: '30', reference_column_name: 'created' },
        'timedelta'
      ),
    ];

    const { edges } = buildGraph(columns);
    expect(edges).toContainEqual({ source: 'a', target: 'b' });
    expect(edges).toContainEqual({ source: 'c', target: 'd' });
  });

  it('flags malformed numeric and JSON sampler params', () => {
    const badNumber = column('a', 'x', 'sampler', { p: 'not-a-number' }, 'bernoulli');
    const badJson = column(
      'b',
      'y',
      'sampler',
      { dist_name: 'beta', dist_params: '{ oops' },
      'scipy'
    );

    expect(validateColumns([badNumber])).toContainEqual(
      expect.stringContaining('must be a number')
    );
    expect(validateColumns([badJson])).toContainEqual(
      expect.stringContaining('must be valid JSON')
    );
  });

  it('accepts a fully-specified numeric sampler as submittable', () => {
    const columns = [column('a', 'trials', 'sampler', { n: '10', p: '0.5' }, 'binomial')];

    expect(validateColumns(columns)).toEqual([]);
  });
});

describe('sampler-showcase template', () => {
  const showcase = FILESET_TEMPLATES.find((t) => t.id === 'sampler-showcase');
  if (!showcase) throw new Error('sampler-showcase template is missing');

  it('covers every previewable sampler sub-type in the palette', () => {
    // The managed `person` sampler is intentionally excluded — it needs downloaded
    // Nemotron Personas datasets and can't preview without them.
    const excluded = new Set<string | undefined>([SamplerType.person]);
    const samplerTypesInPalette = COLUMN_TYPE_GROUPS.flatMap((g) => g.options)
      .filter((o) => o.columnType === 'sampler')
      .map((o) => o.samplerType)
      .filter((t) => !excluded.has(t));
    const covered = new Set(showcase.columns.map((c) => c.samplerType));
    for (const samplerType of samplerTypesInPalette) {
      expect(covered).toContain(samplerType);
    }
    expect(covered.has(SamplerType.person)).toBe(false);
  });

  it('validates clean and builds a config for every sampler', () => {
    const columns = buildColumnsFromTemplate(showcase.columns);
    expect(columns).toHaveLength(showcase.columns.length);
    expect(validateColumns(columns)).toEqual([]);

    const built = buildDataDesignerConfig(columns).columns;
    // Every emitted config is a sampler with a params object.
    for (const config of built) {
      expect(config).toMatchObject({ column_type: 'sampler' });
      expect((config as { params?: unknown }).params).toBeTypeOf('object');
    }
  });

  it('wires the two intended sampler dependency edges', () => {
    const columns = buildColumnsFromTemplate(showcase.columns);
    const { edges } = buildGraph(columns);
    const idOf = (name: string) => columns.find((c) => c.name === name)?.id;

    expect(edges).toContainEqual({
      source: idOf('category_topic'),
      target: idOf('subcategory_topic'),
    });
    expect(edges).toContainEqual({ source: idOf('created_at'), target: idOf('shipped_after') });
  });

  it('coerces representative params to their SDK types', () => {
    const columns = buildColumnsFromTemplate(showcase.columns);
    const built = buildDataDesignerConfig(columns).columns as unknown as Array<{
      sampler_type: string;
      params: Record<string, unknown>;
    }>;
    const paramsFor = (t: SamplerType) => built.find((c) => c.sampler_type === t)?.params;

    expect(paramsFor(SamplerType.uuid)).toMatchObject({ short_form: 'true', uppercase: 'false' });
    expect(paramsFor(SamplerType.category)).toMatchObject({ weights: [3, 2, 1] });
    expect(paramsFor(SamplerType.binomial)).toMatchObject({ n: 10, p: 0.5 });
    expect(paramsFor(SamplerType.scipy)).toMatchObject({ dist_params: { a: 2, b: 5 } });
    expect(paramsFor(SamplerType.timedelta)).toMatchObject({ dt_min: 1, dt_max: 30 });
  });
});

describe('seed-dataset columns', () => {
  it('hoists the fileset source to seed_config and does NOT emit a seed column', () => {
    const columns = [
      column('a', 'seed', 'seed-dataset', {
        fileset_ref: 'default/my-fileset',
        file_path: 'data.parquet',
        sampling_strategy: 'shuffle',
        available_columns: 'question,answer',
      }),
      column('b', 'graded', 'llm-text', {
        prompt: 'Grade {{ answer }}',
        model_alias: 'default',
      }),
    ];

    const config = buildDataDesignerConfig(columns);
    expect(config.columns).toEqual([
      {
        name: 'graded',
        column_type: 'llm-text',
        prompt: 'Grade {{ answer }}',
        model_alias: 'default',
      },
    ]);
    expect(config.seed_config).toEqual({
      source: { seed_type: 'nhx', path: 'default/my-fileset#data.parquet' },
      sampling_strategy: 'shuffle',
    });
  });

  it('omits sampling strategy when unset', () => {
    const columns = [
      column('a', 'seed', 'seed-dataset', {
        fileset_ref: 'default/my-fileset',
        file_path: 'data.parquet',
      }),
    ];

    expect(buildDataDesignerConfig(columns).seed_config).toEqual({
      source: { seed_type: 'nhx', path: 'default/my-fileset#data.parquet' },
    });
  });

  it('omits seed_config until both a fileset and a file are picked', () => {
    const noFile = [column('a', 'seed', 'seed-dataset', { fileset_ref: 'default/my-fileset' })];
    expect(buildDataDesignerConfig(noFile).seed_config).toBeUndefined();

    const none = [column('b', 'seed', 'seed-dataset', {})];
    expect(buildDataDesignerConfig(none).seed_config).toBeUndefined();
  });

  it('requires a fileset and a file before submit', () => {
    const errors = validateColumns([column('a', 'seed', 'seed-dataset', {})]);
    expect(errors).toContainEqual(expect.stringContaining('Fileset is required'));
    expect(errors).toContainEqual(expect.stringContaining('File is required'));
  });

  it('exposes the seed file columns as referenceable outputs and draws edges from the seed node', () => {
    const columns = [
      column('a', 'seed', 'seed-dataset', {
        fileset_ref: 'default/my-fileset',
        file_path: 'data.parquet',
        available_columns: 'question, answer',
      }),
      column('b', 'graded', 'llm-text', {
        prompt: 'Grade the {{ answer }} to {{ question }}',
        model_alias: 'default',
      }),
    ];

    const { nodes, edges } = buildGraph(columns);
    expect(nodes.find((n) => n.id === 'a')?.data.tags).toEqual(['{{question}}', '{{answer}}']);
    expect(edges).toEqual([{ source: 'a', target: 'b' }]);
  });
});

describe('llm-judge columns', () => {
  const scores =
    '[{ "name": "Quality", "description": "Overall answer quality.", "options": { "1": "Very poor", "5": "Excellent" } }]';

  it('serializes required judge score definitions', () => {
    const columns = [
      column('a', 'quality_score', 'llm-judge', {
        prompt: 'Rate {{ answer }}',
        model_alias: 'default',
        scores,
      }),
    ];

    expect(buildDataDesignerConfig(columns).columns[0]).toMatchObject({
      name: 'quality_score',
      column_type: 'llm-judge',
      prompt: 'Rate {{ answer }}',
      model_alias: 'default',
      scores: [
        {
          name: 'Quality',
          description: 'Overall answer quality.',
          options: { '1': 'Very poor', '5': 'Excellent' },
        },
      ],
    });
  });

  it('requires valid score JSON before submit', () => {
    expect(
      validateColumns([
        column('a', 'quality_score', 'llm-judge', {
          prompt: 'Rate {{ answer }}',
          model_alias: 'default',
        }),
      ])
    ).toContainEqual(expect.stringContaining('Scores (JSON) is required'));

    expect(
      validateColumns([
        column('a', 'quality_score', 'llm-judge', {
          prompt: 'Rate {{ answer }}',
          model_alias: 'default',
          scores: '{ bad',
        }),
      ])
    ).toContainEqual(expect.stringContaining('Scores (JSON) must be valid JSON'));
  });
});

describe('preference-pairs template', () => {
  const template = FILESET_TEMPLATES.find((t) => t.id === 'preference-pairs');
  if (!template) throw new Error('preference-pairs template is missing');

  it('builds a valid llm-judge column with scores', () => {
    const columns = buildColumnsFromTemplate(template.columns);

    expect(validateColumns(columns)).toEqual([]);
    expect(buildDataDesignerConfig(columns).columns).toContainEqual(
      expect.objectContaining({
        name: 'quality_score',
        column_type: 'llm-judge',
        scores: [
          expect.objectContaining({
            name: 'Quality',
            options: expect.objectContaining({ '1': 'Very poor', '5': 'Excellent' }),
          }),
        ],
      })
    );
  });
});

describe('topologicalSortColumns', () => {
  const ids = (columns: BuilderColumn[]) => columns.map((c) => c.id);

  it('places each column after the columns it references', () => {
    // Input order deliberately reversed: judge → response → instruction → topic.
    const input = [
      column('c3', 'quality', 'llm-judge', { prompt: 'Rate {{ response }}', model_alias: 'm' }),
      column('c2', 'response', 'llm-text', {
        prompt: 'Answer {{ instruction }}',
        model_alias: 'm',
      }),
      column('c1', 'instruction', 'llm-text', {
        prompt: 'Ask about {{ topic }}',
        model_alias: 'm',
      }),
      column('c0', 'topic', 'sampler', { values: 'a, b' }, SamplerType.category),
    ];
    expect(ids(topologicalSortColumns(input))).toEqual(['c0', 'c1', 'c2', 'c3']);
  });

  it('keeps independent columns and ties in their original relative order', () => {
    const input = [
      column('a', 'alpha', 'sampler', { values: 'x' }, SamplerType.category),
      column('b', 'beta', 'sampler', { values: 'y' }, SamplerType.category),
      column('c', 'gamma', 'sampler', { values: 'z' }, SamplerType.category),
    ];
    expect(ids(topologicalSortColumns(input))).toEqual(['a', 'b', 'c']);
  });

  it('groups by dependency depth, not add order — a shallow dependent sorts above a deeper one', () => {
    // entity_type → description → structured is a depth-2 chain; llm_text_1 depends only on
    // entity_type (depth 1). Even though it was added last, it must sort above `structured`.
    const input = [
      column('c0', 'entity_type', 'sampler', { values: 'a, b' }, SamplerType.category),
      column('c1', 'description', 'llm-text', {
        prompt: 'Describe {{ entity_type }}',
        model_alias: 'm',
      }),
      column('c2', 'structured', 'llm-structured', {
        prompt: 'Structure {{ description }}',
        model_alias: 'm',
        output_format: '{}',
      }),
      column('c3', 'llm_text_1', 'llm-text', {
        prompt: 'Another {{ entity_type }}',
        model_alias: 'm',
      }),
    ];
    // depths: entity_type 0, description 1, llm_text_1 1, structured 2.
    expect(ids(topologicalSortColumns(input))).toEqual(['c0', 'c1', 'c3', 'c2']);
  });

  it('emits cyclic references without looping', () => {
    const input = [
      column('c1', 'one', 'expression', { expr: '{{ two }}' }),
      column('c2', 'two', 'expression', { expr: '{{ one }}' }),
    ];
    expect(ids(topologicalSortColumns(input)).sort()).toEqual(['c1', 'c2']);
  });
});

describe('buildColumnsFromConfig', () => {
  it('round-trips a mixed recipe: config → columns → config is stable', () => {
    const original = [
      column('a', 'domain', 'sampler', { values: 'a, b, c', weights: '3, 1, 1' }, 'category'),
      column('b', 'score', 'sampler', { mean: '1.5', stddev: '0.25' }, 'gaussian'),
      column('c', 'answer', 'llm-text', {
        prompt: 'Answer about {{ domain }}',
        model_alias: 'default',
      }),
      column('d', 'structured', 'llm-structured', {
        prompt: 'Structure {{ answer }}',
        model_alias: 'default',
        output_format: '{ "type": "object" }',
      }),
    ];
    const config = buildDataDesignerConfig(original);

    const rebuilt = buildColumnsFromConfig(config);
    expect(rebuilt.map((c) => c.name)).toEqual(['domain', 'score', 'answer', 'structured']);
    expect(rebuilt.map((c) => c.option.columnType)).toEqual([
      'sampler',
      'sampler',
      'llm-text',
      'llm-structured',
    ]);
    expect(buildDataDesignerConfig(rebuilt)).toEqual(config);
  });

  it('reconstructs the seed-dataset column from seed_config', () => {
    const config = buildDataDesignerConfig([
      column('a', 'seed', 'seed-dataset', {
        fileset_ref: 'default/my-fileset',
        file_path: 'data.parquet',
        sampling_strategy: 'shuffle',
      }),
      column('b', 'graded', 'llm-text', { prompt: 'Grade {{ answer }}', model_alias: 'default' }),
    ]);

    const rebuilt = buildColumnsFromConfig(config);
    const seed = rebuilt.find((c) => c.option.columnType === 'seed-dataset');
    expect(seed?.values).toEqual({
      fileset_ref: 'default/my-fileset',
      file_path: 'data.parquet',
      sampling_strategy: 'shuffle',
    });
    expect(buildDataDesignerConfig(rebuilt).seed_config).toEqual(config.seed_config);
  });

  it('numbers ids from startId and skips column types absent from the palette', () => {
    const config = buildDataDesignerConfig([
      column('a', 'answer', 'llm-text', { prompt: 'Hi', model_alias: 'default' }),
    ]);
    config.columns.push({ name: 'mystery', column_type: 'not-a-type' } as never);

    const rebuilt = buildColumnsFromConfig(config, 5);
    expect(rebuilt).toHaveLength(1);
    expect(rebuilt[0]).toMatchObject({ id: 'col-5', name: 'answer' });
  });
});

describe('palette catalog', () => {
  it('has a field descriptor path for every catalog column type', () => {
    // Sanity: findColumnOption resolves every option the palette can emit.
    for (const group of COLUMN_TYPE_GROUPS) {
      for (const option of group.options) {
        expect(
          findColumnOption({ columnType: option.columnType, samplerType: option.samplerType })
        ).toBe(option);
      }
    }
  });
});
