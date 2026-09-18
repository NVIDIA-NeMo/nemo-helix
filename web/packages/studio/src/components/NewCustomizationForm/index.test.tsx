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
import { FORM_DEFAULTS, type CustomizationFormFields } from '@studio/util/forms/customization';
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
    mutateAutomodel.mockReset().mockResolvedValue(undefined);
    mutateUnsloth.mockReset().mockResolvedValue(undefined);
    mutateRl.mockReset().mockResolvedValue(undefined);
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

    // The banner names the offending field, not a bare Zod message (ASTD-622).
    const banner = await screen.findByText(/Please fix the following errors/i);
    expect(banner.textContent).toMatch(/Model: /);
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

  /**
   * The dataset picker writes the validation reference and no field renders it, so the
   * only thing proving it survives to the API is the request itself.
   */
  it('submits the validation dataset the picker resolved', async () => {
    vi.mocked(useCustomizationDatasetValidation).mockReturnValue({
      ...emptyValidation,
      hasTraining: true,
      hasValidation: true,
    });

    const initialValues: CustomizationFormFields = {
      ...FORM_DEFAULTS,
      backend: 'automodel',
      outputName: 'my-output',
      automodel: {
        ...FORM_DEFAULTS.automodel,
        model: 'default/qwen3-0-6b',
        dataset: { ...FORM_DEFAULTS.automodel.dataset, training: 'default/commonsense_qa' },
      },
    };

    const user = userEvent.setup();
    renderRoute(<NewCustomizationForm workspace="default" initialValues={initialValues} />);

    await user.click(await screen.findByRole('button', { name: /Start Fine-Tuning/i }));

    await waitFor(() => expect(mutateAutomodel).toHaveBeenCalled());
    const [[call]] = mutateAutomodel.mock.calls;
    expect(call.data.spec.dataset).toMatchObject({
      training: 'default/commonsense_qa',
      validation: 'default/commonsense_qa',
    });
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

  it('surfaces schema violations from a seeded config on load, before any submit', async () => {
    // Import paths (clone, template, "start from your own config") set field values directly,
    // bypassing the slider clamps, so an injected 0 against a `.gt(0)` field must be caught on
    // load rather than only at submit.
    const seeded = structuredClone(FORM_DEFAULTS);
    if (seeded.automodel.batch) seeded.automodel.batch.global_batch_size = 0;
    renderRoute(<NewCustomizationForm workspace="default" initialValues={seeded} />);

    const banner = await screen.findByText(/Please fix the following errors/i);
    expect(banner.textContent).toMatch(/Global batch size: Number must be greater than 0/);
  });

  it('clears a seeded validation error once the user corrects the field', async () => {
    const user = userEvent.setup();
    // Output name, model, and dataset are filled (as clone/template seeding does) so the
    // injected batch size is the only schema error, and correcting it clears the whole banner.
    const seeded = structuredClone(FORM_DEFAULTS);
    seeded.outputName = 'my-fine-tune-model';
    seeded.automodel.model = 'meta/llama-3.2-1b-instruct';
    if (seeded.automodel.dataset) seeded.automodel.dataset.training = 'my-training-dataset';
    if (seeded.automodel.batch) seeded.automodel.batch.global_batch_size = 0;
    renderRoute(<NewCustomizationForm workspace="default" initialValues={seeded} />);

    expect(
      await screen.findByText(/Global batch size: Number must be greater than 0/)
    ).toBeInTheDocument();

    // Correcting the field revalidates live and clears the banner without a submit. The slider
    // inputs share an aria-label, so pick this one out by its form field name.
    const input = screen
      .getAllByRole('spinbutton')
      .find((el) => el.getAttribute('name') === 'automodel.batch.global_batch_size');
    if (!(input instanceof HTMLInputElement)) throw new Error('batch size input not found');
    await user.clear(input);
    await user.type(input, '16');

    await waitFor(() =>
      expect(screen.queryByText(/Please fix the following errors/i)).not.toBeInTheDocument()
    );
  });

  it('updates the banner to the remaining error as fields are corrected', async () => {
    const user = userEvent.setup();
    // Two injected errors; correcting one must leave the banner showing only the other.
    const seeded = structuredClone(FORM_DEFAULTS);
    seeded.outputName = 'my-fine-tune-model';
    seeded.automodel.model = 'meta/llama-3.2-1b-instruct';
    if (seeded.automodel.dataset) seeded.automodel.dataset.training = 'my-training-dataset';
    if (seeded.automodel.batch) seeded.automodel.batch.global_batch_size = 0;
    if (seeded.automodel.schedule) seeded.automodel.schedule.epochs = 0;
    renderRoute(<NewCustomizationForm workspace="default" initialValues={seeded} />);

    const banner = await screen.findByText(/Please fix the following errors/i);
    expect(banner.textContent).toMatch(/Global batch size: Number must be greater than 0/);
    expect(banner.textContent).toMatch(/Epochs: Number must be greater than 0/);

    const input = screen
      .getAllByRole('spinbutton')
      .find((el) => el.getAttribute('name') === 'automodel.batch.global_batch_size');
    if (!(input instanceof HTMLInputElement)) throw new Error('batch size input not found');
    await user.clear(input);
    await user.type(input, '16');

    // The corrected field's error drops; the untouched one remains.
    await waitFor(() =>
      expect(screen.getByText(/Please fix the following errors/i).textContent).not.toMatch(
        /Global batch size/
      )
    );
    expect(screen.getByText(/Please fix the following errors/i).textContent).toMatch(
      /Epochs: Number must be greater than 0/
    );
  });
});
