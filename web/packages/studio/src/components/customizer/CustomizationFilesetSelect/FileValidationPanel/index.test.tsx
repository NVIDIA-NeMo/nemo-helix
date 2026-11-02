// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FileValidationPanel } from '@studio/components/customizer/CustomizationFilesetSelect/FileValidationPanel';
import type { CustomizationDatasetValidationResult } from '@studio/hooks/useCustomizationDatasetValidation';
import { CUSTOMIZER_SCHEMA_LABELS } from '@studio/util/customizerSchema';
import { render, screen } from '@testing-library/react';

/**
 * Build a baseline-pass validation result. Individual tests override the one
 * field they care about so each test exercises a single visible branch.
 */
const buildValidation = (
  overrides: Partial<CustomizationDatasetValidationResult> = {}
): CustomizationDatasetValidationResult => ({
  isPending: false,
  discoveryError: null,
  format: { ok: true, fileErrors: [] },
  schema: {
    variant: 'sft-prompt-completion',
    label: CUSTOMIZER_SCHEMA_LABELS['sft-prompt-completion'],
  },
  schemaExpectedCopy: 'Must contain messages (chat) or prompt and completion.',
  schemaMismatchedFiles: [],
  schemaShape: '{\n  prompt: string,\n  completion: string,\n}',
  completeness: { ok: true, skipped: false, errors: [] },
  encoding: { ok: true, fileErrors: [] },
  hasTraining: true,
  hasValidation: true,
  autoSplitNotice: false,
  training: [],
  validation: [],
  trainingRowCount: 100,
  validationRowCount: 20,
  ...overrides,
});

describe('FileValidationPanel', () => {
  describe('overriding states (supersede the checklist)', () => {
    it('renders a load-error banner when discovery itself failed', () => {
      render(
        <FileValidationPanel
          validation={buildValidation({ discoveryError: new Error('Network down') })}
        />
      );
      expect(screen.getByText(/Could not list files in this dataset/)).toBeInTheDocument();
      expect(screen.getByText(/Network down/)).toBeInTheDocument();
      // Checklist label should NOT be rendered.
      expect(screen.queryByText('File Validation')).not.toBeInTheDocument();
    });

    it('renders a spinner while validation is pending', () => {
      render(
        <FileValidationPanel
          validation={buildValidation({ isPending: true, hasTraining: false })}
        />
      );
      expect(screen.getByText('Validating dataset files...')).toBeInTheDocument();
      expect(screen.queryByText('File Validation')).not.toBeInTheDocument();
    });

    it('renders nothing when no training files are present', () => {
      const { container } = render(
        <FileValidationPanel validation={buildValidation({ hasTraining: false })} />
      );
      expect(container).toBeEmptyDOMElement();
    });
  });

  describe('checklist (happy path)', () => {
    it('renders all rows green when every check passes', () => {
      render(<FileValidationPanel validation={buildValidation()} />);
      expect(screen.getByText('File Validation')).toBeInTheDocument();
      expect(screen.getByText(/Format: Single line JSONL is valid/)).toBeInTheDocument();
      expect(
        screen.getByText(`Schema: ${CUSTOMIZER_SCHEMA_LABELS['sft-prompt-completion']}`)
      ).toBeInTheDocument();
      expect(screen.getByText(/Encoding: UTF-8 encoding/)).toBeInTheDocument();
      expect(screen.getByText(/Completeness: No empty or null values/)).toBeInTheDocument();
    });
  });

  describe('format row', () => {
    it('renders sample error message when one or more files failed format/fetch', () => {
      const validation = buildValidation({
        format: {
          ok: false,
          fileErrors: [
            { path: 'training/bad.jsonl', error: 'Failed to download file: 404' },
            { path: 'training/other.jsonl', error: 'parse error at line 3' },
          ],
        },
      });
      render(<FileValidationPanel validation={validation} />);
      expect(
        screen.getByText(/Found 2 files with errors \(e\.g\. training\/bad\.jsonl/)
      ).toBeInTheDocument();
      expect(screen.getByText(/Failed to download file: 404/)).toBeInTheDocument();
    });
  });

  describe('schema row', () => {
    it('renders warning when no file matched any customizer shape', () => {
      const validation = buildValidation({
        schema: null,
        schemaShape: '',
        schemaMismatchedFiles: [],
      });
      render(<FileValidationPanel validation={validation} />);
      expect(
        screen.getByText(
          /Schema: Does not match\. Must contain messages \(chat\) or prompt and completion\./
        )
      ).toBeInTheDocument();
    });

    it('renders fail when some files matched but others did not', () => {
      const validation = buildValidation({
        schemaMismatchedFiles: ['training/odd.jsonl'],
      });
      render(<FileValidationPanel validation={validation} />);
      expect(
        screen.getByText(
          new RegExp(`Schema: ${CUSTOMIZER_SCHEMA_LABELS['sft-prompt-completion']}, but 1 file`)
        )
      ).toBeInTheDocument();
      expect(screen.getByText(/training\/odd\.jsonl/)).toBeInTheDocument();
    });
  });

  describe('encoding row', () => {
    it('renders the offending file path when a single file fails', () => {
      const validation = buildValidation({
        encoding: { ok: false, fileErrors: [{ path: 'training/utf16.jsonl' }] },
      });
      render(<FileValidationPanel validation={validation} />);
      expect(
        screen.getByText(/Encoding: training\/utf16\.jsonl is not valid UTF-8/)
      ).toBeInTheDocument();
    });

    it('renders a count + sample path when multiple files fail', () => {
      const validation = buildValidation({
        encoding: {
          ok: false,
          fileErrors: [{ path: 'training/utf16.jsonl' }, { path: 'validation/utf16.jsonl' }],
        },
      });
      render(<FileValidationPanel validation={validation} />);
      expect(
        screen.getByText(/Encoding: 2 files are not valid UTF-8 \(e\.g\. training\/utf16\.jsonl\)/)
      ).toBeInTheDocument();
    });

    it('falls back to a generic message when encoding.ok is false but fileErrors is empty', () => {
      // Defensive: the hook can in principle produce ok=false with no
      // fileErrors (e.g. perFile.length !== allFiles.length). The panel
      // must not crash on errors[0].
      const validation = buildValidation({
        encoding: { ok: false, fileErrors: [] },
      });
      render(<FileValidationPanel validation={validation} />);
      expect(screen.getByText(/Encoding: UTF-8 encoding check failed/)).toBeInTheDocument();
    });
  });

  describe('completeness row', () => {
    it('renders fail with first offending row', () => {
      const validation = buildValidation({
        completeness: {
          ok: false,
          skipped: false,
          errors: [{ path: 'training/bad.jsonl', row: 3, message: 'rejected is missing or empty' }],
        },
      });
      render(<FileValidationPanel validation={validation} />);
      expect(
        screen.getByText(
          /Completeness: Found 1 row with empty or missing required fields \(e\.g\. training\/bad\.jsonl:3 — rejected is missing or empty\)/
        )
      ).toBeInTheDocument();
    });

    it('hides the completeness row entirely when the check was skipped', () => {
      // We don't know what to require when format failed or schema didn't
      // match, so we drop the row instead of rendering a third ambiguous
      // status. The whole panel keeps a strict pass/fail/warn convention.
      const validation = buildValidation({
        format: { ok: false, fileErrors: [{ path: 'a.jsonl', error: 'bad' }] },
        schema: null,
        schemaShape: '',
        completeness: { ok: false, skipped: true, errors: [] },
      });
      render(<FileValidationPanel validation={validation} />);
      expect(screen.queryByText(/Completeness:/)).not.toBeInTheDocument();
    });

    it('falls back to a generic message when completeness fails with no specific errors', () => {
      // Defensive: shouldn't happen given today's invariants, but guard
      // ensures we never crash on errors[0] if the invariant changes.
      const validation = buildValidation({
        completeness: { ok: false, skipped: false, errors: [] },
      });
      render(<FileValidationPanel validation={validation} />);
      expect(screen.getByText(/Completeness: Completeness check failed/)).toBeInTheDocument();
    });
  });

  describe('auto-split notice', () => {
    it('renders the info banner with split stats when validation is missing', () => {
      const validation = buildValidation({
        hasValidation: false,
        autoSplitNotice: true,
        validationRowCount: 0,
        trainingRowCount: 1000,
      });
      render(<FileValidationPanel validation={validation} />);
      expect(
        screen.getByText('No Validation Data found. Training data will be automatically split.')
      ).toBeInTheDocument();
      expect(screen.getByText(/90% \(900\)/)).toBeInTheDocument();
      expect(screen.getByText(/10% \(100\)/)).toBeInTheDocument();
    });

    it('omits the info banner when validation files are present', () => {
      render(<FileValidationPanel validation={buildValidation()} />);
      expect(
        screen.queryByText('No Validation Data found. Training data will be automatically split.')
      ).not.toBeInTheDocument();
    });
  });

  describe('schema preview', () => {
    it('renders the inferred field shape as a typescript code block', () => {
      const { container } = render(<FileValidationPanel validation={buildValidation()} />);
      expect(screen.getByText('Schema')).toBeInTheDocument();
      // CodeSnippet renders the value into a <pre>/<code> block; check the keys exist.
      expect(container.textContent).toContain('prompt: string');
      expect(container.textContent).toContain('completion: string');
    });

    it('omits the schema block when no shape was inferred', () => {
      render(<FileValidationPanel validation={buildValidation({ schemaShape: '' })} />);
      expect(screen.queryByText('Schema')).not.toBeInTheDocument();
    });
  });
});
