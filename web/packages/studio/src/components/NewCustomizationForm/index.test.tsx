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

const mockReadiness = vi.hoisted(() => vi.fn());
const mockCreateDeploymentConfig = vi.hoisted(() => vi.fn());

vi.mock('@studio/hooks/useBaseModelDeploymentReadiness', () => ({
  useBaseModelDeploymentReadiness: mockReadiness,
}));

vi.mock('@studio/routes/NewDeploymentRoute/useCreateDeploymentBySource', () => ({
  createWorkspaceDeploymentConfig: mockCreateDeploymentConfig,
}));

/** Minimum automodel payload that clears `customizationFormSchema`. */
const validAutomodelValues = (): CustomizationFormFields => ({
  ...FORM_DEFAULTS,
  outputName: 'my-adapter',
  automodel: {
    ...FORM_DEFAULTS.automodel,
    model: 'default/base-model',
    dataset: { training: 'default/my-dataset' },
  },
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
    // `onSubmit` chains `.catch()` onto the mutation, so these must be thenable.
    // Resolved with a job rather than `undefined`: `onSuccess` navigates using
    // `job.name`, which throws when the mutation resolves to nothing.
    mutateAutomodel.mockReset().mockResolvedValue({ name: 'job-1' });
    mutateUnsloth.mockReset().mockResolvedValue({ name: 'job-1' });
    mutateRl.mockReset().mockResolvedValue({ name: 'job-1' });
    mockCreateDeploymentConfig.mockReset();
    mockCreateDeploymentConfig.mockResolvedValue(undefined);
    mockReadiness.mockReset();
    mockReadiness.mockReturnValue({
      state: 'none',
      deploymentName: null,
      status: null,
      isLoading: false,
    });
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

  describe('base model deployment', () => {
    // The section is gated on `producesAdapter`, not on the backend: only an
    // unmerged LoRA run emits an adapter, and only an adapter is served by a
    // deployment of its *base* model.
    it('offers the Deployment section for a LoRA run', async () => {
      renderRoute(<NewCustomizationForm workspace="default" />);
      expect(await screen.findByText('Deployment')).toBeInTheDocument();
    });

    it('hides the Deployment section for lora_merged, whose output is full weights', async () => {
      const values = validAutomodelValues();
      values.automodel.training = {
        ...values.automodel.training,
        finetuning_type: 'lora_merged',
      };
      renderRoute(<NewCustomizationForm workspace="default" initialValues={values} />);

      await screen.findByText('Compute Resources');
      expect(screen.queryByText('Deployment')).not.toBeInTheDocument();
    });

    // Unsloth's merge is a save_method, not a finetuning_type, so `finetuning_type`
    // alone would call this an adapter and offer a base-model deployment for output
    // that is full weights.
    it('hides the Deployment section for a merged unsloth save', async () => {
      const values: CustomizationFormFields = {
        ...FORM_DEFAULTS,
        backend: 'unsloth',
        unsloth: {
          ...FORM_DEFAULTS.unsloth,
          model: { ...FORM_DEFAULTS.unsloth.model, name: 'default/base-model' },
          training: { ...FORM_DEFAULTS.unsloth.training, finetuning_type: 'lora' },
          output: { save_method: 'merged_16bit' },
        },
      };
      renderRoute(<NewCustomizationForm workspace="default" initialValues={values} />);

      await screen.findByText('Compute Resources');
      expect(screen.queryByText('Deployment')).not.toBeInTheDocument();
    });

    it('hides the Deployment section for DPO, which is always full-weight', async () => {
      const values: CustomizationFormFields = { ...FORM_DEFAULTS, backend: 'rl' };
      renderRoute(<NewCustomizationForm workspace="default" initialValues={values} />);

      await screen.findByText('Compute Resources');
      expect(screen.queryByText('Deployment')).not.toBeInTheDocument();
    });

    it('offers no deployment controls when the base already serves LoRA', async () => {
      mockReadiness.mockReturnValue({
        state: 'serving-lora',
        deploymentName: 'base-deployment',
        status: 'READY',
        isLoading: false,
      });
      renderRoute(
        <NewCustomizationForm workspace="default" initialValues={validAutomodelValues()} />
      );

      expect(await screen.findByText(/base-deployment/)).toBeInTheDocument();
      expect(screen.queryByText('Engine')).not.toBeInTheDocument();
    });

    // The config, and only the config. Studio creating the deployment too would
    // reach the same end state hours early and idle a serving GPU for the run.
    it('creates the deployment config before the job and hands the job its name', async () => {
      const user = userEvent.setup();
      renderRoute(
        <NewCustomizationForm workspace="default" initialValues={validAutomodelValues()} />
      );

      await user.click(await screen.findByRole('button', { name: /Start Fine-Tuning/i }));

      await waitFor(() => expect(mutateAutomodel).toHaveBeenCalled());
      // `base-model` derived from the base model ref, plus the wizard's `-config`.
      expect(mockCreateDeploymentConfig).toHaveBeenCalledWith(
        'default',
        expect.anything(),
        'base-model-config',
        expect.any(Function)
      );
      expect(mutateAutomodel).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.objectContaining({
            spec: expect.objectContaining({ deployment_config: 'base-model-config' }),
          }),
        })
      );
      // Ordering is the point: a config that fails must not cost a training run.
      expect(mockCreateDeploymentConfig.mock.invocationCallOrder[0]).toBeLessThan(
        mutateAutomodel.mock.invocationCallOrder[0]
      );
    });

    it('does not start the job when the config is rejected', async () => {
      mockCreateDeploymentConfig.mockRejectedValue(new Error('image pull denied'));
      const user = userEvent.setup();
      renderRoute(
        <NewCustomizationForm workspace="default" initialValues={validAutomodelValues()} />
      );

      await user.click(await screen.findByRole('button', { name: /Start Fine-Tuning/i }));

      expect(await screen.findByText(/image pull denied/i)).toBeInTheDocument();
      await waitFor(() => expect(mutateAutomodel).not.toHaveBeenCalled());
    });

    // Opting out is allowed — the user may have their own serving plan. The job
    // still runs; the section warns the adapter will not be servable until the
    // base model is deployed.
    it('starts the job without deploying when the user opts out', async () => {
      const user = userEvent.setup();
      renderRoute(
        <NewCustomizationForm workspace="default" initialValues={validAutomodelValues()} />
      );

      await user.click(await screen.findByRole('switch', { name: /Deploy the base model/ }));
      await user.click(await screen.findByRole('button', { name: /Start Fine-Tuning/i }));

      await waitFor(() => expect(mutateAutomodel).toHaveBeenCalled());
      expect(mockCreateDeploymentConfig).not.toHaveBeenCalled();
      // No config to point at, so the job must not carry a dangling reference.
      expect(mutateAutomodel.mock.calls[0][0].data.spec.deployment_config).toBeUndefined();
    });

    it('skips the config call when the base already serves LoRA', async () => {
      mockReadiness.mockReturnValue({
        state: 'serving-lora',
        deploymentName: 'base-deployment',
        status: 'READY',
        isLoading: false,
      });
      const user = userEvent.setup();
      renderRoute(
        <NewCustomizationForm workspace="default" initialValues={validAutomodelValues()} />
      );

      await user.click(await screen.findByRole('button', { name: /Start Fine-Tuning/i }));

      await waitFor(() => expect(mutateAutomodel).toHaveBeenCalled());
      expect(mockCreateDeploymentConfig).not.toHaveBeenCalled();
    });
  });
});
