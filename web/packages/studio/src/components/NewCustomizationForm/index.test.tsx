// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// vi.mock calls below are hoisted by vitest, so this import still resolves the mocks.
import { modelsListModels } from '@nemo/sdk/generated/platform/models';
import { NewCustomizationForm } from '@studio/components/NewCustomizationForm';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import {
  CustomizationDatasetValidationResult,
  useCustomizationDatasetValidation,
} from '@studio/hooks/useCustomizationDatasetValidation';
import { mockUseParams } from '@studio/tests/util/mockUseParams';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

const mutateAutomodel = vi.fn();
const mutateUnsloth = vi.fn();
const mutateRl = vi.fn();

vi.mock('@nemo/sdk/generated/customizer/automodel-jobs', () => ({
  useCustomizationCreateAutomodelJob: () => ({ mutateAsync: mutateAutomodel, isPending: false }),
}));

vi.mock('@nemo/sdk/generated/customizer/unsloth-jobs', () => ({
  useCustomizationCreateUnslothJob: () => ({ mutateAsync: mutateUnsloth, isPending: false }),
}));

vi.mock('@nemo/sdk/generated/customizer/rl-jobs', () => ({
  useCustomizationCreateRlJob: () => ({ mutateAsync: mutateRl, isPending: false }),
}));

vi.mock('@nemo/sdk/generated/platform/models', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@nemo/sdk/generated/platform/models')>();
  return { ...actual, modelsListModels: vi.fn() };
});

const mockListModels = vi.mocked(modelsListModels);

vi.mock('@studio/hooks/useCustomizationDatasetValidation', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('@studio/hooks/useCustomizationDatasetValidation')>();
  return { ...actual, useCustomizationDatasetValidation: vi.fn() };
});

const emptyValidation: CustomizationDatasetValidationResult = {
  isPending: false,
  discoveryError: null,
  format: { ok: true, fileErrors: [] },
  schema: null,
  schemaExpectedCopy: '',
  schemaMismatchedFiles: [],
  schemaShape: '',
  completeness: { ok: true, skipped: false, errors: [] },
  encoding: { ok: true, fileErrors: [] },
  hasTraining: false,
  hasValidation: false,
  autoSplitNotice: false,
  training: [],
  validation: [],
  trainingRowCount: 0,
  validationRowCount: 0,
};

describe('NewCustomizationForm', () => {
  beforeEach(() => {
    mutateAutomodel.mockReset();
    mutateUnsloth.mockReset();
    mockUseParams({ [ROUTE_PARAMS.workspace]: 'default' });
    vi.mocked(useCustomizationDatasetValidation).mockReturnValue(emptyValidation);
    mockListModels.mockReset();
    mockListModels.mockResolvedValue({
      data: [],
      pagination: {
        page: 1,
        page_size: 25,
        current_page_size: 0,
        total_results: 0,
        total_pages: 1,
      },
    } as Awaited<ReturnType<typeof modelsListModels>>);
  });

  it('defaults to the automodel backend and shows its compute controls', async () => {
    renderRoute(<NewCustomizationForm workspace="default" />);
    // Automodel exposes multi-GPU parallelism ("GPUs per Node"), not raw indices.
    expect(await screen.findByText('GPUs per Node')).toBeInTheDocument();
    expect(screen.queryByText('GPU Indices')).not.toBeInTheDocument();
  });

  it('swaps to unsloth-specific controls when the unsloth backend is selected', async () => {
    const user = userEvent.setup();
    renderRoute(<NewCustomizationForm workspace="default" />);

    await user.click(await screen.findByRole('radio', { name: /Unsloth/i }));

    // Unsloth exposes single-node GPU indices; automodel parallelism disappears.
    expect(await screen.findByText('GPU Indices')).toBeInTheDocument();
    expect(screen.queryByText('GPUs per Node')).not.toBeInTheDocument();
  });

  /**
   * The backend documents these as mutually exclusive (`load_in_4bit` xor `load_in_8bit`),
   * so the two switches must not both be on. Both off is valid — that is the 16-bit path.
   */
  it('turns off the other quantisation switch when one is enabled', async () => {
    const user = userEvent.setup();
    renderRoute(<NewCustomizationForm workspace="default" />);

    await user.click(await screen.findByRole('radio', { name: /Unsloth/i }));
    // Two sections carry an "Advanced" accordion; the model fields are in the first.
    await user.click((await screen.findAllByText('Advanced'))[0]);

    const fourBit = await screen.findByRole('switch', { name: /Load in 4-bit/i });
    const eightBit = await screen.findByRole('switch', { name: /Load in 8-bit/i });

    // 4-bit is the spec default, so 8-bit starts off.
    expect(fourBit).toBeChecked();
    expect(eightBit).not.toBeChecked();

    await user.click(eightBit);
    expect(eightBit).toBeChecked();
    expect(fourBit).not.toBeChecked();

    await user.click(fourBit);
    expect(fourBit).toBeChecked();
    expect(eightBit).not.toBeChecked();
  });

  /** Both off is the 16-bit path, so turning one off must not switch the other on. */
  it('leaves the other switch alone when one is turned off', async () => {
    const user = userEvent.setup();
    renderRoute(<NewCustomizationForm workspace="default" />);

    await user.click(await screen.findByRole('radio', { name: /Unsloth/i }));
    // Two sections carry an "Advanced" accordion; the model fields are in the first.
    await user.click((await screen.findAllByText('Advanced'))[0]);

    const fourBit = await screen.findByRole('switch', { name: /Load in 4-bit/i });
    const eightBit = await screen.findByRole('switch', { name: /Load in 8-bit/i });

    await user.click(fourBit);

    expect(fourBit).not.toBeChecked();
    expect(eightBit).not.toBeChecked();

    // Each switch clears the other through its own handler, so the 8-bit side needs the
    // same check: turning it on takes 4-bit off, and turning it back off leaves it off.
    await user.click(eightBit);
    expect(eightBit).toBeChecked();
    expect(fourBit).not.toBeChecked();

    await user.click(eightBit);
    expect(eightBit).not.toBeChecked();
    expect(fourBit).not.toBeChecked();
  });

  it('shows the validation banner and does not submit when required fields are missing', async () => {
    const user = userEvent.setup();
    renderRoute(<NewCustomizationForm workspace="default" />);

    await user.click(await screen.findByRole('button', { name: /Start Fine-Tuning/i }));

    expect(await screen.findByText(/Please fix the following errors/i)).toBeInTheDocument();
    expect(mutateAutomodel).not.toHaveBeenCalled();
    expect(mutateUnsloth).not.toHaveBeenCalled();
  });

  it('does not block submit on the inactive backend (only the active one is validated)', async () => {
    // Regression guard: switching to unsloth must not surface stale automodel errors.
    const user = userEvent.setup();
    renderRoute(<NewCustomizationForm workspace="default" />);

    await user.click(await screen.findByRole('radio', { name: /Unsloth/i }));
    await user.click(await screen.findByRole('button', { name: /Start Fine-Tuning/i }));

    // The errors shown must be about the unsloth fields, never automodel ones.
    const banner = await screen.findByText(/Please fix the following errors/i);
    expect(banner.textContent).not.toMatch(/automodel/i);
    await waitFor(() => expect(mutateAutomodel).not.toHaveBeenCalled());
  });

  it('asks the API for fine-tunable models instead of filtering the page client-side', async () => {
    const user = userEvent.setup();
    renderRoute(<NewCustomizationForm workspace="default" />);

    await user.click(await screen.findByTestId('model-select-v2-trigger'));

    await waitFor(() =>
      expect(mockListModels).toHaveBeenCalledWith(
        'default',
        expect.objectContaining({ filter: expect.objectContaining({ fileset: true }) })
      )
    );
  });
});
