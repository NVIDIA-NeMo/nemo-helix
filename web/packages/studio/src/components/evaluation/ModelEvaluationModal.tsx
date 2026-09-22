// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import '@studio/components/evaluation/evalWizard.css';
import { withOperators } from '@nemo/common/src/api/filterOperators';
import { FilesetSearchableSelect } from '@nemo/common/src/components/FilesetSearchableSelect';
import { FormModal, type FormModalProps } from '@nemo/common/src/components/FormModal';
import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { useFilesListFilesets } from '@nemo/sdk/generated/platform/files';
import {
  Banner,
  Button,
  Anchor,
  Flex,
  FormField,
  RadioGroup,
  Stack,
  Stepper,
  Text,
  Tooltip,
} from '@nvidia/foundations-react-core';
import { MODEL_EVAL_CONFIG_DESCRIPTION } from '@studio/components/evaluation/submitEvaluationJob';
import { LINK_EVAL_DOCS } from '@studio/constants/links';
import { ConfigureStep } from '@studio/routes/evaluation/EvaluationNewRoute/ConfigureStep';
import { EvaluationStep } from '@studio/routes/evaluation/EvaluationNewRoute/EvaluationStep';
import { specToFormValues } from '@studio/routes/evaluation/EvaluationNewRoute/specToFormValues';
import {
  COMPARISON_METRICS,
  EVALUATION_FORM_DEFAULTS,
  type EvaluationFormValues,
  evaluationSchema,
  NEW_CONFIG,
  REFERENCE_METRICS,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { useCreateEvaluation } from '@studio/routes/evaluation/EvaluationNewRoute/useCreateEvaluation';
import { useDatasetBindings } from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetBindings';
import {
  type SavedConfig,
  useSavedConfig,
} from '@studio/routes/evaluation/EvaluationNewRoute/useSavedConfig';
import { Info } from 'lucide-react';
import { type FC, useEffect, useState } from 'react';
import { FormProvider, useForm, useFormContext, useWatch } from 'react-hook-form';

/** Radio value for the reuse branch. Not a config name, so it never collides
 *  with one: `configSource` only holds a fileset name once one is picked. */
const EXISTING_CONFIG = 'existing';

type WizardStep = 'start' | 'configuration' | 'evaluation';

/** Re-using a configuration skips authoring one -- the same way the agent
 *  wizard's re-run path skips creating an experiment. */
const stepsFor = (reusing: boolean): WizardStep[] =>
  reusing ? ['start', 'evaluation'] : ['start', 'configuration', 'evaluation'];
const STEP_HEADINGS: Record<WizardStep, string> = {
  start: 'Begin',
  configuration: 'Configuration',
  evaluation: 'Run Evaluation',
};

/** The bare fileset name: `configSource` is passed to endpoints that take a
 *  name, and the picker is already workspace-scoped. */
const configOption = (fileset: { name: string }) => ({
  value: fileset.name,
  label: fileset.name,
});

const startItems = (rerunDisabled: boolean) => [
  {
    value: NEW_CONFIG,
    children: (
      <Stack gap="density-xs">
        <Text kind="label/bold/md">Create a new evaluation configuration</Text>
        <Text kind="body/regular/sm" color="secondary">
          Choose a dataset, map its fields, and pick the metrics to score every row with. The
          configuration is saved so later runs can reuse it.
        </Text>
      </Stack>
    ),
  },
  {
    value: EXISTING_CONFIG,
    disabled: rerunDisabled,
    children: (
      <Stack gap="density-xs">
        <Flex align="center" gap="density-xs">
          <Text kind="label/bold/md">Re-run an existing evaluation</Text>
          {rerunDisabled && (
            <Tooltip slotContent="No saved evaluation configurations to re-run.">
              <Info size={14} aria-label="Why is this option disabled?" />
            </Tooltip>
          )}
        </Flex>
        <Text kind="body/regular/sm" color="secondary">
          Reuses the configuration saved on a previous run. Everything but the model is fixed, so
          the two runs stay comparable.
        </Text>
      </Stack>
    ),
  },
];

/** The first screen: which way in. */
const StartStep: FC<{
  workspace: string;
  mode: string;
  onModeChange: (mode: string) => void;
}> = ({ workspace, mode, onModeChange }) => {
  const { setValue, reset, getValues } = useFormContext<EvaluationFormValues>();

  // One row is enough to answer "is there anything to re-run?". Until it
  // resolves the option stays enabled, so a slow list never disables a choice
  // that turns out to be available.
  const savedConfigs = useFilesListFilesets(workspace, {
    page_size: 1,
    filter: withOperators({ description: { $eq: MODEL_EVAL_CONFIG_DESCRIPTION } }),
  });
  const noSavedConfigs = !savedConfigs.isLoading && (savedConfigs.data?.data?.length ?? 0) === 0;

  return (
    <Stack gap="density-lg">
      <FormField slotLabel="How do you want to start?">
        <RadioGroup
          kind="tile"
          orientation="vertical"
          name="configMode"
          items={startItems(noSavedConfigs)}
          value={mode}
          onValueChange={(value) => {
            onModeChange(value);
            if (value === NEW_CONFIG) {
              // Authoring starts from nothing, so a configuration loaded before
              // backtracking here cannot leak into it. The model is chosen per
              // run and survives.
              const { model } = getValues();
              reset({ ...EVALUATION_FORM_DEFAULTS, configSource: NEW_CONFIG, model });
            } else {
              setValue('configSource', '', { shouldValidate: false });
            }
          }}
        />
      </FormField>
    </Stack>
  );
};

const ConfigSourceField: FC<{ workspace: string }> = ({ workspace }) => {
  const {
    control,
    formState: { errors },
  } = useFormContext<EvaluationFormValues>();

  return (
    <FilesetSearchableSelect<EvaluationFormValues>
      workspace={workspace}
      useControllerProps={{ name: 'configSource', control }}
      formFieldProps={{
        slotLabel: 'Evaluation Configuration',
        slotError: errors.configSource?.message,
      }}
      description={MODEL_EVAL_CONFIG_DESCRIPTION}
      renderOption={configOption}
      triggerPlaceholder="Select a saved configuration"
    />
  );
};

/** Seeds the form from the chosen config. A reused config is shown in the real
 *  form, so every control has to be filled from what was saved. */
const SavedConfigLoader: FC<{
  filesetName: string;
  workspace: string;
  saved: SavedConfig;
}> = ({ filesetName, workspace, saved }) => {
  const { reset, getValues } = useFormContext<EvaluationFormValues>();
  const { spec, isLoading, error } = saved;

  useEffect(() => {
    const { model } = getValues();
    reset({ ...EVALUATION_FORM_DEFAULTS, configSource: filesetName, model });
  }, [filesetName, reset, getValues]);

  useEffect(() => {
    if (!spec) return;
    const current = getValues();
    reset({
      ...current,
      ...specToFormValues(spec, workspace),
      configSource: filesetName,
      // The model is chosen per run, so a config swap must not clear one the
      // user already picked.
      model: current.model,
    });
  }, [spec, filesetName, workspace, reset, getValues]);

  if (isLoading) return <Text kind="body/regular/md">Loading configuration...</Text>;
  if (error)
    return (
      <Banner kind="inline" status="error">
        {error}
      </Banner>
    );
  return null;
};

const ModalBody: FC<{
  workspace: string;
  step: WizardStep;
  steps: WizardStep[];
  mode: string;
  onModeChange: (mode: string) => void;
  saved: SavedConfig;
}> = ({ workspace, step, steps, mode, onModeChange, saved }) => {
  const { control } = useFormContext<EvaluationFormValues>();
  const configSource = useWatch({ control, name: 'configSource' });
  const reusing = mode === EXISTING_CONFIG;

  return (
    <Stack gap="density-2xl">
      <Stepper
        className="eval-wizard-stepper"
        aria-label="New evaluation progress"
        activeStep={steps.indexOf(step)}
        items={steps.map((item) => ({ slotHeading: STEP_HEADINGS[item] }))}
      />
      {step === 'start' ? (
        <>
          <Text kind="body/regular/md">
            Run evaluation via NeMo Evaluator&apos;s built-in runner.{' '}
            <Anchor
              kind="inline"
              textKind="body/regular/md"
              href={LINK_EVAL_DOCS}
              target="_blank"
              rel="noreferrer"
            >
              Learn more
            </Anchor>
            .
          </Text>
          <StartStep workspace={workspace} mode={mode} onModeChange={onModeChange} />
        </>
      ) : null}
      {step === 'configuration' ? <ConfigureStep /> : null}
      {step === 'evaluation' ? (
        <Stack gap="density-lg">
          {reusing ? <ConfigSourceField workspace={workspace} /> : null}
          {/* Seeds the form from the saved config. Nothing is shown back to the
              user -- the configuration step is skipped -- but the Live Test and
              the submitted spec are both built from these values. */}
          {reusing && configSource ? (
            <SavedConfigLoader filesetName={configSource} workspace={workspace} saved={saved} />
          ) : null}
          {!reusing || saved.spec ? <EvaluationStep /> : null}
        </Stack>
      ) : null}
    </Stack>
  );
};

export interface ModelEvaluationModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  workspace: string;
}

/**
 * Run Model Evaluation, as a two-step wizard.
 *
 * The first step decides whether this run authors a configuration or reuses a
 * saved one; the second is the configuration itself, inert when reused. Wider
 * than the agent modal because that second step is a three-column layout.
 */
export const ModelEvaluationModal: FC<ModelEvaluationModalProps> = ({
  open,
  onClose,
  workspace,
}) => {
  const [step, setStep] = useState<WizardStep>('start');
  /** Wizard navigation, not part of the spec: known before a configuration is
   *  named, so it cannot live on `configSource`. */
  const [mode, setMode] = useState<string>(NEW_CONFIG);
  const form = useForm<EvaluationFormValues>({
    defaultValues: EVALUATION_FORM_DEFAULTS,
    resolver: zodResolver(evaluationSchema),
    mode: 'onChange',
  });
  const close = () => {
    form.reset(EVALUATION_FORM_DEFAULTS);
    setStep('start');
    setMode(NEW_CONFIG);
    onClose();
  };

  return (
    <FormProvider {...form}>
      <ModelEvaluationModalInner
        open={open}
        onClose={close}
        workspace={workspace}
        step={step}
        setStep={setStep}
        mode={mode}
        onModeChange={setMode}
      />
    </FormProvider>
  );
};

/** Split out so the submit handler can read the bindings, which are derived from
 *  form state and therefore only available inside the provider. */
const ModelEvaluationModalInner: FC<{
  open: boolean;
  onClose: () => void;
  workspace: string;
  step: WizardStep;
  setStep: (step: WizardStep) => void;
  mode: string;
  onModeChange: (mode: string) => void;
}> = ({ open, onClose, workspace, step, setStep, mode, onModeChange }) => {
  const form = useFormContext<EvaluationFormValues>();
  const bindings = useDatasetBindings();
  const { createEvaluation, isPending } = useCreateEvaluation();
  const configSource = useWatch({ control: form.control, name: 'configSource' });

  const reusing = mode === EXISTING_CONFIG;
  const saved = useSavedConfig(reusing && configSource ? configSource : null);
  const metrics = useWatch({ control: form.control, name: 'body.metrics' });

  /** A messages configuration stores no input or reference path -- both are
   *  array-indexed, so `toFieldMapping` drops them -- and `useMessagesBinding`
   *  only re-derives them once the preview row lands. Submitting in that window
   *  fails validation on a field this step does not render. */
  const needsReference = [...REFERENCE_METRICS, ...COMPARISON_METRICS].some(
    (metric) => metrics?.[metric]
  );
  const bindingsReady =
    Boolean(bindings.inputPath) && (!needsReference || Boolean(bindings.referencePath));
  const configReady = !reusing || (Boolean(saved.spec) && bindingsReady);

  const steps = stepsFor(reusing);
  const at = steps.indexOf(step);
  const isLast = at === steps.length - 1;

  const handleSubmit = form.handleSubmit(
    (values) => {
      void createEvaluation(values, bindings);
    },
    (errors) => {
      // Every field except the model lives on an earlier screen, so a failure
      // here would otherwise mark fields the user cannot see. Send them back to
      // the step that owns the first error.
      const onEarlierStep = ['name', 'dataset', 'fieldMapping', 'body'].some(
        (field) => field in errors
      );
      if (onEarlierStep && steps.includes('configuration')) setStep('configuration');
    }
  );

  /** What each step is answerable for. Validating the whole schema here would
   *  refuse to advance over a field on a later step -- the model is not chosen
   *  until the last one. */
  const FIELDS_BY_STEP: Record<WizardStep, (keyof EvaluationFormValues)[]> = {
    start: [],
    configuration: ['name', 'dataset', 'fieldMapping', 'body'],
    evaluation: ['model', 'configSource'],
  };

  /** Next validates what is on screen, so errors land on the fields that own
   *  them rather than arriving as a summary. */
  const goNext = async () => {
    const fields = FIELDS_BY_STEP[step];
    if (fields.length > 0 && !(await form.trigger(fields))) return;
    setStep(steps[Math.min(at + 1, steps.length - 1)]);
  };

  const goBack = () => setStep(steps[Math.max(at - 1, 0)]);

  return (
    <FormModal
      open={open}
      onClose={onClose}
      title="Run Model Evaluation"
      // Required by the props even though `slotFooterRight` replaces the button
      // it labels; the agent modal passes it for the same reason.
      submitButtonText="Submit"
      disabled={isPending}
      // One width for both steps, matching the agent modal: the configuration
      // step is a single column now, so it no longer needs the extra room.
      className="w-[690px]! max-w-[95vw]!"
      onSubmit={handleSubmit}
      // The whole footer, matching the agent modal: Cancel, then Back, then the
      // forward action. Supplying this replaces FormModal's default pair, which
      // is the only way to get Back between them rather than off to the left.
      slotFooterRight={
        <Flex gap="2">
          <Button kind="tertiary" type="button" onClick={onClose} disabled={isPending}>
            Cancel
          </Button>
          {at > 0 && (
            <Button kind="secondary" type="button" onClick={goBack} disabled={isPending}>
              Back
            </Button>
          )}
          {isLast ? (
            <LoadingButton
              color="brand"
              type="submit"
              loading={isPending}
              disabled={isPending || !configReady}
            >
              Submit
            </LoadingButton>
          ) : (
            <Button color="brand" type="button" onClick={() => void goNext()} disabled={isPending}>
              Next
            </Button>
          )}
        </Flex>
      }
    >
      <ModalBody
        workspace={workspace}
        step={step}
        steps={steps}
        mode={mode}
        onModeChange={onModeChange}
        saved={saved}
      />
    </FormModal>
  );
};
