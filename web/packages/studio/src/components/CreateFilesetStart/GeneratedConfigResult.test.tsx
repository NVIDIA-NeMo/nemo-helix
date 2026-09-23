// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { GeneratedConfigResult } from '@studio/components/CreateFilesetStart/GeneratedConfigResult';
import type { GeneratedConfigValidation } from '@studio/routes/DataDesignerJobBuildRoute/aiSeed';
import { render, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

const RAW_OUTPUT = '{\n  "job_request": {\n    "name": "qa-pairs"\n  }\n}';

const INVALID: GeneratedConfigValidation = {
  status: 'invalid',
  errors: ['The generated config has no columns.'],
  warnings: [],
};

const VALID_WITH_WARNING: GeneratedConfigValidation = {
  status: 'valid',
  jobRequest: { name: 'qa-pairs', spec: { num_records: 25, config: { columns: [] } } } as never,
  seed: { name: 'qa-pairs', rows: '25', columns: [], models: [] },
  warnings: ['Skipped 1 column(s) the builder can’t edit: thumbnail (image).'],
};

const renderResult = (props: Partial<Parameters<typeof GeneratedConfigResult>[0]> = {}) =>
  render(
    <GeneratedConfigResult
      validation={INVALID}
      requestError={null}
      rawOutput={RAW_OUTPUT}
      isGenerating={false}
      isFixing={false}
      {...props}
    />
  );

describe('GeneratedConfigResult', () => {
  it('opens the raw config in a side panel from View config', async () => {
    const user = userEvent.setup();
    renderResult();

    expect(screen.queryByText(/"qa-pairs"/)).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /view config/i }));

    expect(await screen.findByText(/Generated config/)).toBeInTheDocument();
    expect(screen.getByText(/"qa-pairs"/)).toBeInTheDocument();
  });

  it('offers View config for a rejected draft, alongside the errors', () => {
    renderResult();

    expect(screen.getByText('The generated config has no columns.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /view config/i })).toBeInTheDocument();
  });

  it('hides View config when there is no output to show', () => {
    renderResult({ rawOutput: null });

    expect(screen.queryByRole('button', { name: /view config/i })).not.toBeInTheDocument();
  });

  describe('fix', () => {
    it('offers a fix for warnings on an otherwise loadable draft', async () => {
      const user = userEvent.setup();
      const onFix = vi.fn();
      renderResult({ validation: VALID_WITH_WARNING, onFix });

      await user.click(screen.getByRole('button', { name: /fix these warnings/i }));

      expect(onFix).toHaveBeenCalledTimes(1);
    });

    it('offers a fix for the errors that block a rejected draft', async () => {
      const user = userEvent.setup();
      const onFix = vi.fn();
      renderResult({ onFix });

      await user.click(screen.getByRole('button', { name: /fix these errors/i }));

      expect(onFix).toHaveBeenCalledTimes(1);
      // An invalid draft sends its errors and warnings together, so no second button.
      expect(screen.queryByRole('button', { name: /fix these warnings/i })).not.toBeInTheDocument();
    });

    it('hides the fix affordance when there is no draft to send back', () => {
      renderResult({ validation: VALID_WITH_WARNING, rawOutput: null, onFix: vi.fn() });

      expect(screen.queryByRole('button', { name: /fix these/i })).not.toBeInTheDocument();
    });

    it('replaces the result with progress copy while the model reworks the draft', () => {
      renderResult({ isFixing: true, onFix: vi.fn() });

      expect(screen.getByText('Working through the issues…')).toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /fix these/i })).not.toBeInTheDocument();
    });
  });
});
