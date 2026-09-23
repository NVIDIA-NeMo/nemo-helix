// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Deployments are a preview flag, off by default. Stubbed explicitly rather than left
// to the default so this file keeps testing the flag-off wizard if that default ever
// flips. Its own file because the flag is read once, when the module graph loads.
vi.hoisted(() => {
  vi.stubEnv('VITE_FF_DEPLOYMENTS_ENABLED', 'false');
});

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

vi.mock('@nemo/sdk/generated/customizer/automodel-jobs', () => ({
  useCustomizationCreateAutomodelJob: () => ({ mutateAsync: mutateAutomodel, isPending: false }),
}));

vi.mock('@nemo/sdk/generated/customizer/unsloth-jobs', () => ({
  useCustomizationCreateUnslothJob: () => ({ mutateAsync: vi.fn(), isPending: false }),
}));

vi.mock('@nemo/sdk/generated/customizer/rl-jobs', () => ({
  useCustomizationCreateRlJob: () => ({ mutateAsync: vi.fn(), isPending: false }),
}));

vi.mock('@nemo/sdk/generated/platform/models', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@nemo/sdk/generated/platform/models')>();
  return { ...actual, modelsListModels: vi.fn().mockResolvedValue({ data: [], pagination: {} }) };
});

vi.mock('@studio/hooks/useCustomizationDatasetValidation', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('@studio/hooks/useCustomizationDatasetValidation')>();
  return { ...actual, useCustomizationDatasetValidation: vi.fn() };
});

const mockReadiness = vi.hoisted(() => vi.fn());
const mockCreateDeploymentConfig = vi.hoisted(() => vi.fn());
const mockCreateUnboundConfig = vi.hoisted(() => vi.fn());

vi.mock('@studio/hooks/useBaseModelDeploymentReadiness', () => ({
  useBaseModelDeploymentReadiness: mockReadiness,
}));

vi.mock('@studio/routes/NewDeploymentRoute/useCreateDeploymentBySource', () => ({
  ensureWorkspaceDeploymentConfig: mockCreateDeploymentConfig,
  ensureUnboundDeploymentConfig: mockCreateUnboundConfig,
}));

const emptyValidation = {
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
} as CustomizationDatasetValidationResult;

/** A LoRA run — the adapter flow, whose section is the one with a readiness query. */
const adapterValues = (): CustomizationFormFields => ({
  ...FORM_DEFAULTS,
  outputName: 'my-adapter',
  automodel: {
    ...FORM_DEFAULTS.automodel,
    model: 'default/base-model',
    dataset: { training: 'default/my-dataset' },
  },
});

/** A full-weight run, whose target is the model the run itself emits. */
const fullWeightValues = (): CustomizationFormFields => {
  const values = adapterValues();
  values.outputName = 'my-model';
  values.automodel.training = { ...values.automodel.training, finetuning_type: 'all_weights' };
  return values;
};

describe('NewCustomizationForm with deployments disabled', () => {
  beforeEach(() => {
    mutateAutomodel.mockReset().mockResolvedValue({ name: 'job-1' });
    mockCreateDeploymentConfig.mockReset();
    mockCreateUnboundConfig.mockReset();
    mockReadiness.mockReset().mockReturnValue({
      state: 'none',
      deploymentName: null,
      status: null,
      isLoading: false,
    });
    mockUseParams({ [ROUTE_PARAMS.workspace]: 'default' });
    vi.mocked(useCustomizationDatasetValidation).mockReturnValue(emptyValidation);
  });

  it.each([
    ['adapter', adapterValues],
    ['full-weight', fullWeightValues],
  ])('offers no Deployment section for a %s run', async (_label, values) => {
    renderRoute(<NewCustomizationForm workspace="default" initialValues={values()} />);

    // Waits on a section that is always present, so the absence below is checked
    // against a rendered form rather than an empty one.
    expect(await screen.findByText('Compute Resources')).toBeInTheDocument();
    expect(screen.queryByText('Deployment')).not.toBeInTheDocument();
    expect(screen.queryByRole('switch', { name: /Deploy/ })).not.toBeInTheDocument();
  });

  it('never asks whether the base model is already serving', async () => {
    renderRoute(<NewCustomizationForm workspace="default" initialValues={adapterValues()} />);

    await screen.findByText('Compute Resources');
    expect(mockReadiness).toHaveBeenCalledWith(expect.anything(), { enabled: false });
  });

  it.each([
    ['adapter', adapterValues],
    ['full-weight', fullWeightValues],
  ])('starts a %s job with no deployment config', async (_label, values) => {
    const user = userEvent.setup();
    renderRoute(<NewCustomizationForm workspace="default" initialValues={values()} />);

    await user.click(await screen.findByRole('button', { name: /Start Fine-Tuning/i }));

    await waitFor(() => expect(mutateAutomodel).toHaveBeenCalled());
    expect(mockCreateDeploymentConfig).not.toHaveBeenCalled();
    expect(mockCreateUnboundConfig).not.toHaveBeenCalled();
    expect(mutateAutomodel.mock.calls[0][0].data.spec.deployment_config).toBeUndefined();
  });

  // A cloned job can carry a `deployment_config` from the run it was cloned from.
  // With deployments off there is nothing to manage what it would create, so it is
  // dropped — same as an explicit opt-out with the flag on.
  it('drops a cloned deployment_config', async () => {
    const user = userEvent.setup();
    const values = adapterValues();
    (values.automodel as { deployment_config?: string }).deployment_config = 'stale-config';
    renderRoute(<NewCustomizationForm workspace="default" initialValues={values} />);

    await user.click(await screen.findByRole('button', { name: /Start Fine-Tuning/i }));

    await waitFor(() => expect(mutateAutomodel).toHaveBeenCalled());
    expect(mutateAutomodel.mock.calls[0][0].data.spec.deployment_config).toBeUndefined();
  });
});
