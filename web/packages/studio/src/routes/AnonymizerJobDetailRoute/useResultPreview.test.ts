// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useResultPreview } from '@studio/routes/AnonymizerJobDetailRoute/useResultPreview';
import { renderHook } from '@testing-library/react';

const fileContentByPath: Record<string, string> = {
  'metadata.json': JSON.stringify({ original_text_column: 'text' }),
  'trace.parquet': [
    JSON.stringify({
      text: 'Cheryl Gray lives in Illinois.',
      text_replaced: 'Sofia Nguyen lives in Colorado.',
      _detected_entities: {
        entities: [{ value: 'Cheryl Gray', label: 'PERSON', start_position: 0, end_position: 11 }],
      },
      final_entities: {
        entities: [{ value: 'Cheryl Gray', label: 'PERSON', start_position: 0, end_position: 11 }],
      },
      _replacement_map: {
        replacements: [{ original: 'Cheryl Gray', label: 'PERSON', synthetic: 'Sofia Nguyen' }],
      },
    }),
  ].join('\n'),
};

const pendingPaths = new Set<string>();

vi.mock('@studio/api/datasets/useDatasetFileContent', () => ({
  useDatasetFileContent: ({ path }: { path: string }) => {
    const key = Object.keys(fileContentByPath).find((suffix) => path.endsWith(suffix));
    const isLoading = !!key && pendingPaths.has(key);
    return {
      data: !key || isLoading ? undefined : fileContentByPath[key],
      isLoading,
      error: null,
    };
  },
}));

const renderPreview = () =>
  renderHook(() => useResultPreview('workspace', 'fileset/job#results/attempt-1/artifacts'));

describe('useResultPreview', () => {
  beforeEach(() => pendingPaths.clear());

  it('reads rows and their entity columns from the trace artifact', () => {
    const { result } = renderPreview();

    expect(result.current.rows).toHaveLength(1);
    expect(result.current.rows[0].text).toBe('Cheryl Gray lives in Illinois.');
    expect(result.current.rows[0].text_replaced).toBe('Sofia Nguyen lives in Colorado.');
    expect(result.current.rows[0]._replacement_map).toEqual({
      replacements: [{ original: 'Cheryl Gray', label: 'PERSON', synthetic: 'Sofia Nguyen' }],
    });
    expect(result.current.rows[0].final_entities).toBeDefined();
  });

  it('resolves the text column from metadata', () => {
    expect(renderPreview().result.current.textColumn).toBe('text');
  });

  it('stays loading while the trace artifact is in flight', () => {
    pendingPaths.add('trace.parquet');

    const { result } = renderPreview();

    expect(result.current.isLoading).toBe(true);
    expect(result.current.rows).toHaveLength(0);
  });

  it('stays loading while metadata is in flight so no row renders without its text column', () => {
    pendingPaths.add('metadata.json');

    const { result } = renderPreview();

    expect(result.current.isLoading).toBe(true);
  });
});
