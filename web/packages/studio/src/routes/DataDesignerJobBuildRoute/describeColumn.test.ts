// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SamplerType } from '@nemo/sdk/generated/data-designer/schema/SamplerType';
import type { ColumnTypeOption } from '@studio/components/AddColumnPalette/types';
import {
  type BuilderColumn,
  findColumnOption,
} from '@studio/routes/DataDesignerJobBuildRoute/columns';
import { describeColumn } from '@studio/routes/DataDesignerJobBuildRoute/describeColumn';

const optionFor = (columnType: string, samplerType?: string): ColumnTypeOption => {
  const found = findColumnOption({ columnType, samplerType } as never);
  if (!found) throw new Error(`no palette option for ${columnType}/${samplerType}`);
  return found;
};

const column = (
  columnType: string,
  values: Record<string, string>,
  samplerType?: string
): BuilderColumn => ({
  id: 'col-0',
  name: 'col',
  option: optionFor(columnType, samplerType),
  values,
});

describe('describeColumn', () => {
  it('labels the badge with the palette option label', () => {
    expect(describeColumn(column('llm-text', {})).typeLabel).toBe('LLM-Text');
    expect(describeColumn(column('sampler', {}, SamplerType.category)).typeLabel).toBe('Category');
  });

  it('joins a few category values with slashes and counts larger sets', () => {
    expect(
      describeColumn(column('sampler', { values: 'easy, medium, hard' }, SamplerType.category))
        .detail
    ).toBe('easy / medium / hard');
    expect(
      describeColumn(
        column('sampler', { values: 'CUDA, Triton, TensorRT, cuDNN, NCCL' }, SamplerType.category)
      ).detail
    ).toBe('5 values · CUDA, Triton, TensorRT, cuDNN, …');
  });

  it('summarizes an llm-judge column from its parsed scores', () => {
    const scores = JSON.stringify([
      { name: 'relevance' },
      { name: 'accuracy' },
      { name: 'fluency' },
    ]);
    expect(describeColumn(column('llm-judge', { scores })).detail).toBe(
      'judge · 3 scores: relevance, accuracy, fluency'
    );
  });

  it('falls back to a bare label when judge scores are missing or malformed', () => {
    expect(describeColumn(column('llm-judge', {})).detail).toBe('judge');
    expect(describeColumn(column('llm-judge', { scores: 'not json' })).detail).toBe('judge');
  });

  it('describes generators and transforms', () => {
    expect(describeColumn(column('llm-text', {})).detail).toBe('generator');
    expect(describeColumn(column('llm-code', { code_lang: 'python' })).detail).toBe(
      'code generator · python'
    );
    expect(describeColumn(column('expression', {})).detail).toBe('expression · no LLM');
  });
});
