// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CustomizationFilesetSelect } from '@studio/components/customizer/CustomizationFilesetSelect';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import {
  CustomizationDatasetValidationResult,
  useCustomizationDatasetValidation,
} from '@studio/hooks/useCustomizationDatasetValidation';
import { datasets } from '@studio/mocks/datasets';
import { mockUseParams } from '@studio/tests/util/mockUseParams';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import type { CustomizerSchemaVariant } from '@studio/util/customizerSchema';
import { FORM_DEFAULTS, type CustomizationFormFields } from '@studio/util/forms/customization';
import userEvent from '@testing-library/user-event';
import { FC, ReactNode } from 'react';
import { FieldPath, FormProvider, useForm, useWatch } from 'react-hook-form';

vi.mock('@studio/hooks/useCustomizationDatasetValidation', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('@studio/hooks/useCustomizationDatasetValidation')>();
  return { ...actual, useCustomizationDatasetValidation: vi.fn() };
});

const mockValidation = vi.mocked(useCustomizationDatasetValidation);

const buildValidation = (
  overrides: Partial<CustomizationDatasetValidationResult> = {}
): CustomizationDatasetValidationResult => ({
  isPending: false,
  discoveryError: null,
  format: { ok: true, fileErrors: [] },
  schema: { variant: 'sft-prompt-completion', label: 'SFT prompt/completion' },
  schemaExpectedCopy: 'Must contain messages (chat) or prompt and completion.',
  schemaMismatchedFiles: [],
  schemaShape: '',
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

const withVariant = (variant: CustomizerSchemaVariant): CustomizationDatasetValidationResult =>
  buildValidation({ schema: { variant, label: variant } });

/** Renders a form field's current value so tests can assert what the picker wrote. */
const FieldSpy: FC<{ name: FieldPath<CustomizationFormFields> }> = ({ name }) => {
  const value = useWatch<CustomizationFormFields>({ name });
  return <div data-testid={`spy:${name}`}>{String(value ?? '')}</div>;
};

/** Wraps the picker in a real RHF context seeded from the production defaults. */
const Harness: FC<{
  children: ReactNode;
  overrides?: Partial<CustomizationFormFields>;
}> = ({ children, overrides }) => {
  const methods = useForm<CustomizationFormFields>({
    defaultValues: { ...FORM_DEFAULTS, ...overrides },
  });
  return <FormProvider {...methods}>{children}</FormProvider>;
};

const firstFileset = datasets.data[0]!;
const firstRef = `${firstFileset.workspace}/${firstFileset.name}`;

describe('CustomizationFilesetSelect', () => {
  beforeEach(() => {
    mockUseParams({ [ROUTE_PARAMS.workspace]: 'default' });
    mockValidation.mockReturnValue(buildValidation());
  });

  it('writes the picked fileset reference into automodel.dataset.training', async () => {
    const user = userEvent.setup();
    renderRoute(
      <Harness overrides={{ backend: 'automodel' }}>
        <CustomizationFilesetSelect />
        <FieldSpy name="automodel.dataset.training" />
      </Harness>
    );

    const trigger = await screen.findByRole('combobox', { name: /dataset/i });
    await user.click(trigger);
    await user.click(await screen.findByRole('option', { name: firstFileset.name ?? '' }));

    await waitFor(() =>
      expect(screen.getByTestId('spy:automodel.dataset.training')).toHaveTextContent(firstRef)
    );
  });

  /**
   * Automodel and unsloth take a second reference for validation, and the canonical job
   * JSON points both at the same fileset. Without this the picker filled only the training
   * reference, so a dataset carrying validation rows trained without them.
   */
  it('mirrors the fileset into automodel.dataset.validation when it has validation files', async () => {
    const user = userEvent.setup();
    renderRoute(
      <Harness overrides={{ backend: 'automodel' }}>
        <CustomizationFilesetSelect />
        <FieldSpy name="automodel.dataset.validation" />
      </Harness>
    );

    await user.click(await screen.findByRole('combobox', { name: /dataset/i }));
    await user.click(await screen.findByRole('option', { name: firstFileset.name ?? '' }));

    await waitFor(() =>
      expect(screen.getByTestId('spy:automodel.dataset.validation')).toHaveTextContent(firstRef)
    );
  });

  it('mirrors the fileset into unsloth.dataset.validation_path', async () => {
    const user = userEvent.setup();
    renderRoute(
      <Harness overrides={{ backend: 'unsloth' }}>
        <CustomizationFilesetSelect />
        <FieldSpy name="unsloth.dataset.validation_path" />
      </Harness>
    );

    await user.click(await screen.findByRole('combobox', { name: /dataset/i }));
    await user.click(await screen.findByRole('option', { name: firstFileset.name ?? '' }));

    await waitFor(() =>
      expect(screen.getByTestId('spy:unsloth.dataset.validation_path')).toHaveTextContent(firstRef)
    );
  });

  /** No validation files means the backend should apply its own 90/10 split instead. */
  it('leaves the validation reference unset when the fileset has none', async () => {
    mockValidation.mockReturnValue(
      buildValidation({ hasValidation: false, autoSplitNotice: true, validationRowCount: 0 })
    );
    const user = userEvent.setup();
    renderRoute(
      <Harness overrides={{ backend: 'automodel' }}>
        <CustomizationFilesetSelect />
        <FieldSpy name="automodel.dataset.training" />
        <FieldSpy name="automodel.dataset.validation" />
      </Harness>
    );

    await user.click(await screen.findByRole('combobox', { name: /dataset/i }));
    await user.click(await screen.findByRole('option', { name: firstFileset.name ?? '' }));

    await waitFor(() =>
      expect(screen.getByTestId('spy:automodel.dataset.training')).toHaveTextContent(firstRef)
    );
    expect(screen.getByTestId('spy:automodel.dataset.validation')).toHaveTextContent('');
  });

  /**
   * Discovery reports no files both while it runs and when it fails, so neither state is
   * evidence that the fileset lacks validation rows. The field is editable by hand, so
   * writing on that non-evidence would also wipe what the user typed.
   */
  it.each([
    ['discovery is still running', { isPending: true, hasValidation: false }],
    ['discovery failed', { discoveryError: new Error('boom'), hasValidation: false }],
  ])('leaves a hand-entered validation reference alone while %s', async (_label, overrides) => {
    mockValidation.mockReturnValue(buildValidation(overrides));
    const user = userEvent.setup();
    renderRoute(
      <Harness
        overrides={{
          backend: 'automodel',
          automodel: {
            ...FORM_DEFAULTS.automodel,
            dataset: { ...FORM_DEFAULTS.automodel.dataset, validation: 'default/typed-by-hand' },
          },
        }}
      >
        <CustomizationFilesetSelect />
        <FieldSpy name="automodel.dataset.validation" />
      </Harness>
    );

    await user.click(await screen.findByRole('combobox', { name: /dataset/i }));
    await user.click(await screen.findByRole('option', { name: firstFileset.name ?? '' }));

    await waitFor(() =>
      expect(screen.getByTestId('spy:automodel.dataset.validation')).toHaveTextContent(
        'default/typed-by-hand'
      )
    );
  });

  it('writes the picked fileset reference into unsloth.dataset.path', async () => {
    const user = userEvent.setup();
    renderRoute(
      <Harness overrides={{ backend: 'unsloth' }}>
        <CustomizationFilesetSelect />
        <FieldSpy name="unsloth.dataset.path" />
      </Harness>
    );

    const trigger = await screen.findByRole('combobox', { name: /dataset/i });
    await user.click(trigger);
    await user.click(await screen.findByRole('option', { name: firstFileset.name ?? '' }));

    await waitFor(() =>
      expect(screen.getByTestId('spy:unsloth.dataset.path')).toHaveTextContent(firstRef)
    );
  });

  it('opens the create-fileset modal when New Dataset is selected', async () => {
    const user = userEvent.setup();
    renderRoute(
      <Harness overrides={{ backend: 'automodel' }}>
        <CustomizationFilesetSelect />
      </Harness>
    );

    const trigger = await screen.findByRole('combobox', { name: /dataset/i });
    await user.click(trigger);
    await user.click(await screen.findByRole('option', { name: 'New Dataset' }));

    expect(await screen.findByText('Create New Dataset')).toBeInTheDocument();
  });

  it('surfaces the no-training-files error when the selected dataset has none', async () => {
    mockValidation.mockReturnValue(buildValidation({ hasTraining: false }));
    renderRoute(
      <Harness
        overrides={{
          backend: 'automodel',
          automodel: { ...FORM_DEFAULTS.automodel, dataset: { training: firstRef } },
        }}
      >
        <CustomizationFilesetSelect />
      </Harness>
    );

    expect(
      await screen.findByText(/No training files were found in this dataset/)
    ).toBeInTheDocument();
  });

  describe('apply_chat_template auto-detection (unsloth)', () => {
    it('enables the chat template for chat/messages datasets', async () => {
      mockValidation.mockReturnValue(withVariant('sft-chat'));
      renderRoute(
        <Harness
          overrides={{
            backend: 'unsloth',
            unsloth: {
              ...FORM_DEFAULTS.unsloth,
              dataset: {
                ...FORM_DEFAULTS.unsloth.dataset,
                path: firstRef,
                apply_chat_template: false,
              },
            },
          }}
        >
          <CustomizationFilesetSelect />
          <FieldSpy name="unsloth.dataset.apply_chat_template" />
        </Harness>
      );

      await waitFor(() =>
        expect(screen.getByTestId('spy:unsloth.dataset.apply_chat_template')).toHaveTextContent(
          'true'
        )
      );
    });

    it('disables the chat template for prompt-completion datasets', async () => {
      mockValidation.mockReturnValue(withVariant('sft-prompt-completion'));
      renderRoute(
        <Harness
          overrides={{
            backend: 'unsloth',
            unsloth: {
              ...FORM_DEFAULTS.unsloth,
              dataset: {
                ...FORM_DEFAULTS.unsloth.dataset,
                path: firstRef,
                apply_chat_template: true,
              },
            },
          }}
        >
          <CustomizationFilesetSelect />
          <FieldSpy name="unsloth.dataset.apply_chat_template" />
        </Harness>
      );

      await waitFor(() =>
        expect(screen.getByTestId('spy:unsloth.dataset.apply_chat_template')).toHaveTextContent(
          'false'
        )
      );
    });
  });
});
