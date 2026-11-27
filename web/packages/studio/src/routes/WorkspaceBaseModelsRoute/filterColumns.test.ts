// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  isSupportedFilter,
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
        lora: 'adapters',
        family: ' llama ',
        model_providers: 'default/nvidia-build',
      })
    ).toEqual({
      fileset: true,
      adapters: true,
      family: { $like: 'llama' },
      model_providers: 'default/nvidia-build',
    });
  });

  it('maps Serving enabled to lora_enabled', () => {
    expect(toModelEntityFilter({ lora: 'lora_enabled' })).toEqual({ lora_enabled: true });
  });

  it('adds the name search as a substring match', () => {
    expect(toModelEntityFilter(undefined, ' qwen ')).toEqual({ name: { $like: 'qwen' } });
  });

  it('ignores Fine-tunable and Has adapters while Customizer is disabled', () => {
    mockEnvironment.customizerEnabled = false;
    expect(
      toModelEntityFilter({ customizable: { fine_tunable: true }, lora: 'adapters' })
    ).toBeUndefined();
  });
});

describe('isSupportedFilter', () => {
  it('accepts offered LoRA options and other filters', () => {
    expect(isSupportedFilter({ id: 'lora', value: 'adapters' })).toBe(true);
    expect(isSupportedFilter({ id: 'customizable', value: { fine_tunable: true } })).toBe(true);
    expect(isSupportedFilter({ id: 'family', value: 'llama' })).toBe(true);
  });

  it('rejects LoRA values that are not offered options', () => {
    expect(isSupportedFilter({ id: 'lora', value: { adapters: true } })).toBe(false);
    expect(isSupportedFilter({ id: 'lora', value: 'unknown' })).toBe(false);
  });

  it('rejects Customizer-only filters while Customizer is disabled', () => {
    mockEnvironment.customizerEnabled = false;
    expect(isSupportedFilter({ id: 'customizable', value: { fine_tunable: true } })).toBe(false);
    expect(isSupportedFilter({ id: 'lora', value: 'adapters' })).toBe(false);
    expect(isSupportedFilter({ id: 'lora', value: 'lora_enabled' })).toBe(true);
  });
});

describe('makeFilterColumns', () => {
  // Stub accessor returns the column def, so the result is the list of defs.
  const getColumns = () =>
    (
      makeFilterColumns('ws') as unknown as (helper: {
        accessor: (_: unknown, def: ColumnDef) => ColumnDef;
      }) => ColumnDef[]
    )({ accessor: (_, def) => def });

  interface ColumnDef {
    id: string;
    enableColumnFilter?: boolean;
  }

  it('orders filters to match the Models API spec', () => {
    expect(getColumns().map((col) => col.id)).toEqual([
      'customizable',
      'lora',
      'family',
      'model_providers',
      'created_at',
      'updated_at',
    ]);
  });

  it('keeps Fine-tunable out of the filter panel, leaving it to the toolbar', () => {
    expect(getColumns().find((col) => col.id === 'customizable')?.enableColumnFilter).toBe(false);
    mockEnvironment.customizerEnabled = false;
    expect(getColumns().map((col) => col.id)).not.toContain('customizable');
  });
});
