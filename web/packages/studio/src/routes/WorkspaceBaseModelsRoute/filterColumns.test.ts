// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  makeFilterColumns,
  toModelEntityFilter,
} from '@studio/routes/WorkspaceBaseModelsRoute/filterColumns';

const mockEnvironment = vi.hoisted(() => ({ customizerEnabled: true }));

vi.mock('@studio/constants/environment', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/constants/environment')>()),
  get CUSTOMIZER_ENABLED() {
    return mockEnvironment.customizerEnabled;
  },
}));

beforeEach(() => {
  mockEnvironment.customizerEnabled = true;
});

describe('toModelEntityFilter', () => {
  it('returns undefined without filters', () => {
    expect(toModelEntityFilter(undefined)).toBeUndefined();
    expect(toModelEntityFilter({ family: '  ' })).toBeUndefined();
  });

  it('maps column filters to the Models API filter', () => {
    expect(
      toModelEntityFilter({
        customizable: { fine_tunable: true },
        lora: { adapters: true, lora_enabled: true },
        family: ' llama ',
        model_providers: 'default/nvidia-build',
      })
    ).toEqual({
      fileset: true,
      adapters: true,
      lora_enabled: true,
      family: { $like: 'llama' },
      model_providers: 'default/nvidia-build',
    });
  });

  it('adds the name search as a substring match', () => {
    expect(toModelEntityFilter(undefined, ' qwen ')).toEqual({ name: { $like: 'qwen' } });
  });

  it('ignores Fine-tunable and Has adapters while Customizer is disabled', () => {
    mockEnvironment.customizerEnabled = false;
    expect(
      toModelEntityFilter({
        customizable: { fine_tunable: true },
        lora: { adapters: true, lora_enabled: true },
      })
    ).toEqual({ lora_enabled: true });
  });
});

describe('makeFilterColumns', () => {
  it('leaves Fine-tunable to the toolbar checkbox, not the filter menu', () => {
    // Stub accessor returns the column def, so the result is the list of defs.
    const makeColumns = makeFilterColumns('ws') as unknown as (helper: {
      accessor: (_: unknown, def: { id: string }) => { id: string };
    }) => { id: string }[];
    const ids = makeColumns({ accessor: (_, def) => def }).map((col) => col.id);
    expect(ids).toContain('family');
    expect(ids).not.toContain('customizable');
  });
});
