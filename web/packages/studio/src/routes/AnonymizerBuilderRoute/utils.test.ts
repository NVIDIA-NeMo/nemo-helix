// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DataDesignerModelOption } from '@studio/components/NewDataDesignerJobForm/utils';
import {
  detectorOptions,
  isNerDetectorModel,
  outputHeadingForStrategy,
  tabForValidationErrors,
} from '@studio/routes/AnonymizerBuilderRoute/utils';

const model = (name: string, servedModelName: string, id = name): DataDesignerModelOption =>
  ({ id, name, served_model_name: servedModelName }) as DataDesignerModelOption;

const option = (value: string) => ({ label: value, value });

describe('isNerDetectorModel', () => {
  it('matches on either identifier, regardless of case', () => {
    expect(isNerDetectorModel(model('nvidia-gliner-pii', 'nvidia/gliner-PII'))).toBe(true);
    expect(isNerDetectorModel(model('pii-detector', 'nvidia/GLiNER-pii'))).toBe(true);
    expect(isNerDetectorModel(model('nvidia-gliner-pii', ''))).toBe(true);
  });

  it('matches privacy filters however they are punctuated', () => {
    expect(isNerDetectorModel(model('privacy-filter', 'openai/privacy-filter'))).toBe(true);
    expect(isNerDetectorModel(model('openai-privacy_filter', ''))).toBe(true);
    expect(isNerDetectorModel(model('OpenAI Privacy Filter', ''))).toBe(true);
  });

  it('ignores the entity id, whose first segment is the workspace', () => {
    expect(
      isNerDetectorModel(model('pii-detector', 'nvidia/nemotron', 'gliner-team/pii-detector'))
    ).toBe(false);
  });

  it('does not match general chat models', () => {
    expect(
      isNerDetectorModel(model('nemotron-3-nano-30b-a3b', 'nvidia/nemotron-3-nano-30b-a3b'))
    ).toBe(false);
    expect(isNerDetectorModel(model('gpt-oss-120b', 'openai/gpt-oss-120b'))).toBe(false);
  });
});

describe('detectorOptions', () => {
  const models = [
    model('nemotron-3-nano-30b-a3b', 'nvidia/nemotron-3-nano-30b-a3b'),
    model('nvidia-gliner-pii', 'nvidia/gliner-PII'),
    model('privacy-filter', 'openai/privacy-filter'),
  ];
  const items = models.map((m) => option(m.id));

  it('keeps every model selectable', () => {
    expect(detectorOptions(items, models).map((item) => item.value)).toEqual(
      expect.arrayContaining(items.map((item) => item.value))
    );
  });

  it('suggests every known NER family, not only GLiNER', () => {
    const grouped = detectorOptions(items, models);
    expect(grouped.slice(0, 2)).toMatchObject([
      { value: 'nvidia-gliner-pii', group: 'suggested' },
      { value: 'privacy-filter', group: 'suggested' },
    ]);
    expect(grouped[2]).toMatchObject({ value: 'nemotron-3-nano-30b-a3b', group: 'other' });
  });

  it('preserves the incoming order within a group', () => {
    const twoGliners = [...models, model('gliner-small', 'nvidia/gliner-small')];
    expect(
      detectorOptions(
        twoGliners.map((m) => option(m.id)),
        twoGliners
      ).map((item) => item.value)
    ).toEqual(['nvidia-gliner-pii', 'privacy-filter', 'gliner-small', 'nemotron-3-nano-30b-a3b']);
  });

  it('returns every model as Other when no known detector is registered', () => {
    const chatOnly = models.filter((m) => !isNerDetectorModel(m));
    const grouped = detectorOptions(
      chatOnly.map((m) => option(m.id)),
      chatOnly
    );
    expect(grouped.map((item) => item.group)).toEqual(['other']);
  });
});

describe('tabForValidationErrors', () => {
  it('stays on Source whenever a Source field failed', () => {
    expect(tabForValidationErrors(['source'])).toBe('source');
    expect(tabForValidationErrors(['source', 'roleModels'])).toBe('source');
    expect(tabForValidationErrors(['entityLabels', 'roleModels'])).toBe('source');
  });

  it('switches to Model Settings only when models are the sole failure', () => {
    expect(tabForValidationErrors(['roleModels'])).toBe('model-settings');
  });

  it('stays on Source when no fields are reported', () => {
    expect(tabForValidationErrors([])).toBe('source');
  });
});

describe('outputHeadingForStrategy', () => {
  it('names the rewrite output', () => {
    expect(outputHeadingForStrategy('rewrite')).toBe('Rewritten');
  });

  it('names the replace output for every other strategy', () => {
    expect(outputHeadingForStrategy('substitute')).toBe('Replaced');
    expect(outputHeadingForStrategy('redact')).toBe('Replaced');
    expect(outputHeadingForStrategy('annotate')).toBe('Replaced');
    expect(outputHeadingForStrategy('hash')).toBe('Replaced');
  });
});
