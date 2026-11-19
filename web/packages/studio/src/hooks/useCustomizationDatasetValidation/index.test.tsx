// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import {
  type CustomizationDatasetValidationResult,
  useCustomizationDatasetValidation,
} from '@studio/hooks/useCustomizationDatasetValidation';
import { server } from '@studio/mocks/node';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import type { FC } from 'react';

/**
 * Shape we serialize from the hook into the DOM. JSON-stringified so each
 * test asserts only the field it cares about with regex / textContent checks.
 */
interface HarnessSnapshot {
  isPending: boolean;
  formatOk: boolean;
  formatFileErrorPaths: string[];
  schemaVariant: string | null;
  schemaMismatchedFiles: string[];
  encodingOk: boolean;
  encodingFileErrorPaths: string[];
  completenessOk: boolean;
  completenessSkipped: boolean;
  completenessErrorCount: number;
  trainingRowCount: number;
  validationRowCount: number;
  hasTraining: boolean;
}

const snapshot = (r: CustomizationDatasetValidationResult): HarnessSnapshot => ({
  isPending: r.isPending,
  formatOk: r.format.ok,
  formatFileErrorPaths: r.format.fileErrors.map((e) => e.path),
  schemaVariant: r.schema?.variant ?? null,
  schemaMismatchedFiles: r.schemaMismatchedFiles,
  encodingOk: r.encoding.ok,
  encodingFileErrorPaths: r.encoding.fileErrors.map((e) => e.path),
  completenessOk: r.completeness.ok,
  completenessSkipped: r.completeness.skipped,
  completenessErrorCount: r.completeness.errors.length,
  trainingRowCount: r.trainingRowCount,
  validationRowCount: r.validationRowCount,
  hasTraining: r.hasTraining,
});

interface HarnessProps {
  fileset: string;
  trainingType: 'sft' | 'dpo';
  sampleLimit?: number;
}

const HookHarness: FC<HarnessProps> = ({ fileset, trainingType, sampleLimit }) => {
  const result = useCustomizationDatasetValidation({ fileset, trainingType, sampleLimit });
  return <span data-testid="snapshot">{JSON.stringify(snapshot(result))}</span>;
};

/** Helper: read the latest snapshot from the harness DOM node. */
const readSnapshot = (): HarnessSnapshot =>
  JSON.parse(screen.getByTestId('snapshot').textContent ?? '{}');

/** Wait until the validation has resolved (isPending === false). */
const waitForResolved = async (): Promise<HarnessSnapshot> => {
  await waitFor(() => {
    expect(readSnapshot().isPending).toBe(false);
  });
  return readSnapshot();
};

/**
 * Build the four msw overrides needed to drive a single fileset under test:
 *  - list filesets (the validation hook doesn't call this; fileset retrieval does)
 *  - list fileset files (returns the discovered file paths)
 *  - HEAD per file (200)
 *  - GET (download) per file (returns the content for that specific path)
 *
 * `contents` maps file path -> UTF-8 string body returned by the GET handler.
 */
const setupFilesetMock = (
  workspace: string,
  name: string,
  files: { path: string; size: number }[],
  contents: Record<string, string>
) => {
  const filesetUrl = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/${workspace}/filesets/${name}`;
  server.use(
    http.get(`${filesetUrl}/files`, () =>
      HttpResponse.json({
        data: files.map((f) => ({ ...f, file_ref: `ref-${f.path}` })),
        pagination: {
          page: 1,
          page_size: 100,
          current_page_size: files.length,
          total_pages: 1,
          total_results: files.length,
        },
      })
    ),
    http.head(`${filesetUrl}/-/*`, () => new HttpResponse(null, { status: 200 })),
    http.get(`${filesetUrl}/-/*`, ({ request }) => {
      const url = new URL(request.url);
      // Path is everything after /-/
      const match = url.pathname.match(/\/-\/(.+)$/);
      const filePath = match ? decodeURIComponent(match[1]) : '';
      const body = contents[filePath];
      if (body === undefined) return new HttpResponse(null, { status: 404 });
      return new HttpResponse(body, {
        status: 200,
        headers: { 'Content-Type': 'application/octet-stream' },
      });
    })
  );
};

const FILESET_URN = 'fileset://test-workspace/test-fileset';

describe('useCustomizationDatasetValidation', () => {
  describe('happy path', () => {
    it('format-pass + schema=sft-prompt-completion when content is well-formed', async () => {
      const lines = Array.from({ length: 5 }, (_, i) =>
        JSON.stringify({ prompt: `q${i}`, completion: `a${i}` })
      ).join('\n');
      setupFilesetMock(
        'test-workspace',
        'test-fileset',
        [{ path: 'training/data.jsonl', size: lines.length }],
        { 'training/data.jsonl': lines }
      );

      render(
        <TestProviders>
          <HookHarness fileset={FILESET_URN} trainingType="sft" />
        </TestProviders>
      );

      const result = await waitForResolved();
      expect(result.formatOk).toBe(true);
      expect(result.schemaVariant).toBe('sft-prompt-completion');
      expect(result.schemaMismatchedFiles).toEqual([]);
      expect(result.completenessOk).toBe(true);
      expect(result.encodingOk).toBe(true);
      expect(result.trainingRowCount).toBe(5);
      expect(result.hasTraining).toBe(true);
    });

    it('detects schema=sft-chat when rows have a messages array', async () => {
      const line = JSON.stringify({
        messages: [
          { role: 'user', content: 'hi' },
          { role: 'assistant', content: 'hello' },
        ],
      });
      setupFilesetMock(
        'test-workspace',
        'test-fileset',
        [{ path: 'training/chat.jsonl', size: line.length }],
        { 'training/chat.jsonl': `${line}\n${line}` }
      );

      render(
        <TestProviders>
          <HookHarness fileset={FILESET_URN} trainingType="sft" />
        </TestProviders>
      );

      const result = await waitForResolved();
      expect(result.schemaVariant).toBe('sft-chat');
    });
  });

  describe('schema-not-detected', () => {
    it('returns null schema and skips completeness when no SFT shape matches', async () => {
      const line = JSON.stringify({ foo: 'bar', baz: 'qux' });
      setupFilesetMock(
        'test-workspace',
        'test-fileset',
        [{ path: 'training/odd.jsonl', size: line.length }],
        { 'training/odd.jsonl': `${line}\n${line}` }
      );

      render(
        <TestProviders>
          <HookHarness fileset={FILESET_URN} trainingType="sft" />
        </TestProviders>
      );

      const result = await waitForResolved();
      expect(result.schemaVariant).toBeNull();
      // Schema mismatched files lists files where format passed but schema didn't.
      expect(result.schemaMismatchedFiles).toEqual(['training/odd.jsonl']);
      expect(result.completenessSkipped).toBe(true);
    });
  });

  describe('sampleLimit', () => {
    /**
     * Build content with 60 valid prompt-completion rows and one malformed
     * line at row index 60 (the 61st line). Format check should:
     *   - sampleLimit=0 (default): scan everything → fail
     *   - sampleLimit=10: only scan first 10 lines → pass
     * This pins that sampleLimit actually bounds the scan and that the
     * default really walks the full file.
     */
    const buildContentWithMalformedPastFifty = () => {
      const valid = Array.from({ length: 60 }, (_, i) =>
        JSON.stringify({ prompt: `q${i}`, completion: `a${i}` })
      );
      const malformed = '{not valid json';
      return [...valid, malformed].join('\n');
    };

    it('default sampleLimit=0 walks the full file and catches a malformed row past 50', async () => {
      const content = buildContentWithMalformedPastFifty();
      setupFilesetMock(
        'test-workspace',
        'test-fileset',
        [{ path: 'training/data.jsonl', size: content.length }],
        { 'training/data.jsonl': content }
      );

      render(
        <TestProviders>
          <HookHarness fileset={FILESET_URN} trainingType="sft" />
        </TestProviders>
      );

      const result = await waitForResolved();
      expect(result.formatOk).toBe(false);
      expect(result.formatFileErrorPaths).toEqual(['training/data.jsonl']);
    });

    it('sampleLimit=10 truncates the scan so the malformed past-50 row is not reached', async () => {
      const content = buildContentWithMalformedPastFifty();
      setupFilesetMock(
        'test-workspace',
        'test-fileset',
        [{ path: 'training/data.jsonl', size: content.length }],
        { 'training/data.jsonl': content }
      );

      render(
        <TestProviders>
          <HookHarness fileset={FILESET_URN} trainingType="sft" sampleLimit={10} />
        </TestProviders>
      );

      const result = await waitForResolved();
      expect(result.formatOk).toBe(true);
      expect(result.schemaVariant).toBe('sft-prompt-completion');
      // Row count still reflects the full file, not the sample.
      expect(result.trainingRowCount).toBe(61);
    });
  });

  // Encoding test (non-UTF-8 input flagged as failure) intentionally omitted
  // here: jsdom + blob-polyfill don't preserve raw binary bytes through Blob/
  // arrayBuffer round-trips, so a non-UTF-8 fixture decodes as if it were
  // valid. The encoding query itself uses `TextDecoder('utf-8', { fatal: true })`
  // — a standard browser API — and is verified manually with the UTF-16 BOM
  // fixture in tmp/training-utf16.jsonl.
});
