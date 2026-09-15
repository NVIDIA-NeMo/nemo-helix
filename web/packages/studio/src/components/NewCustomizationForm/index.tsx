// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { generateDefaultName } from '@nemo/common/src/utils/generateDefaultName';
import { useCustomizationCreateAutomodelJob } from '@nemo/sdk/generated/customizer/automodel-jobs';
import { useCustomizationCreateRlJob } from '@nemo/sdk/generated/customizer/rl-jobs';
import { useCustomizationCreateUnslothJob } from '@nemo/sdk/generated/customizer/unsloth-jobs';
import {
  Banner,
  Button,
  Divider,
  Flex,
  PageHeader,
  Panel,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { CustomizationFilesetSelect } from '@studio/components/customizer/CustomizationFilesetSelect';
import { BackendSelectionSection } from '@studio/components/NewCustomizationForm/BackendSelectionSection';
import {
  baseDeploymentDefaults,
  baseDeploymentName,
  DEPLOY_BY_DEFAULT,
  outputDeploymentDefaults,
} from '@studio/components/NewCustomizationForm/baseDeploymentForm';
import { ComputeResourcesSection } from '@studio/components/NewCustomizationForm/ComputeResourcesSection';
import { DeploymentSection } from '@studio/components/NewCustomizationForm/DeploymentSection';
import { DpoParametersSection } from '@studio/components/NewCustomizationForm/DpoParametersSection';
import { GeneralParametersSection } from '@studio/components/NewCustomizationForm/GeneralParametersSection';
import { GrpoParametersSection } from '@studio/components/NewCustomizationForm/GrpoParametersSection';
import { IntegrationsSection } from '@studio/components/NewCustomizationForm/IntegrationsSection';
import { LoraParametersSection } from '@studio/components/NewCustomizationForm/LoraParametersSection';
import { ModelSelectionSection } from '@studio/components/NewCustomizationForm/ModelSelectionSection';
import { OutputDeploymentSection } from '@studio/components/NewCustomizationForm/OutputDeploymentSection';
import { RewardEnvironmentSection } from '@studio/components/NewCustomizationForm/RewardEnvironmentSection';
import { TrainingMethodSection } from '@studio/components/NewCustomizationForm/TrainingMethodSection';
import {
  useBaseModelDeploymentReadiness,
  type BaseModelDeploymentState,
} from '@studio/hooks/useBaseModelDeploymentReadiness';
import {
  configNameFromWizardBaseName,
  createDeploymentWizardSchema,
  type WizardFormValues,
} from '@studio/routes/NewDeploymentRoute/schema';
import { ensureWorkspaceDeploymentConfig } from '@studio/routes/NewDeploymentRoute/useCreateDeploymentBySource';
import { getWorkspaceCustomizationJobDetailsRoute } from '@studio/routes/utils';
import {
  FORM_DEFAULTS,
  customizationFormSchema,
  formToAutomodelCreate,
  formToRlCreate,
  formToUnslothCreate,
  MODEL_FIELD_BY_BACKEND,
  producesAdapter,
  type CustomizationFormFields,
} from '@studio/util/forms/customization';
import { FC, useEffect, useMemo, useRef, useState } from 'react';
import { type FieldErrors, FormProvider, type Resolver, useForm, useWatch } from 'react-hook-form';
import { useNavigate } from 'react-router';

/**
 * Turn a react-hook-form field path into a human label for the error banner, so a bare
 * Zod message ("Number must be greater than 0") says which field it belongs to. Generalizes
 * across every field and every Zod error: the leaf path segment, minus array indices, spaced
 * and capitalized.
 *
 * ponytail: derived from the field path, not the UI slotLabel — `batch_size` reads
 * "Batch size", not the control's "Global Batch Size". Swap in a path->label map here if
 * exact UI labels are ever required.
 */
const humanizeFieldPath = (path: string): string => {
  const leaf = path
    .split('.')
    .filter((segment) => segment && !/^\d+$/.test(segment))
    .pop();
  if (!leaf) return '';
  const spaced = leaf.replace(/_/g, ' ').trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
};

/**
 * Flatten a react-hook-form error tree into deduped, field-labeled messages for the banner.
 * Used by submit validation; the load-time check uses `messagesFromZodIssues`. Both share
 * `humanizeFieldPath`, so an imported config and a rejected submit read identically.
 * Returns [] when there are no errors.
 */
const collectErrorMessages = (errors: FieldErrors<CustomizationFormFields>): string[] => {
  const messages: string[] = [];
  const walk = (node: unknown, path: string) => {
    if (!node || typeof node !== 'object') return;
    if ('message' in node && typeof node.message === 'string') {
      const label = humanizeFieldPath(path);
      messages.push(label ? `${label}: ${node.message}` : node.message);
      return;
    }
    Object.entries(node).forEach(([key, value]) => walk(value, path ? `${path}.${key}` : key));
  };
  walk(errors, '');
  return Array.from(new Set(messages));
};

/** Same field-labeled banner messages as `collectErrorMessages`, but from a Zod parse (load). */
const messagesFromZodIssues = (
  issues: readonly { path: readonly (string | number)[]; message: string }[]
): string[] =>
  Array.from(
    new Set(
      issues.map((issue) => {
        const label = humanizeFieldPath(issue.path.join('.'));
        return label ? `${label}: ${issue.message}` : issue.message;
      })
    )
  );

/**
 * Readiness states in which creating a base-model deployment is the right move.
 *
 * An allowlist rather than `!== 'serving-lora'`, because the negative form treats every
 * state it does not name as grounds to create a deployment — which is how
 * `'indeterminate'`, the state that means "we could not find out", would become the state
 * that acts. Any state added later gets the safe default of doing nothing.
 */
const CREATE_DEPLOYMENT_STATES: readonly BaseModelDeploymentState[] = [
  'none',
  'serving-without-lora',
  'unavailable',
];

interface NewCustomizationFormProps {
  workspace: string;
  initialModel?: string;
  initialValues?: CustomizationFormFields;
}

export const NewCustomizationForm: FC<NewCustomizationFormProps> = ({
  workspace,
  initialModel,
  initialValues,
}) => {
  const navigate = useNavigate();
  const toast = useToast();
  const errorBannerRef = useRef<HTMLDivElement>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [deployStage, setDeployStage] = useState<string | null>(null);
  // Two pieces of state rather than one, despite the shared default: the sections
  // ask about different targets, so an opt-out for one should not silently carry
  // over when the training method changes and back.
  const [deployBaseModel, setDeployBaseModel] = useState(DEPLOY_BY_DEFAULT);
  const [deployOutputModel, setDeployOutputModel] = useState(DEPLOY_BY_DEFAULT);

  const defaultValues = useMemo<CustomizationFormFields>(() => {
    if (initialValues) return initialValues;
    return {
      ...FORM_DEFAULTS,
      outputName: generateDefaultName(),
      automodel: { ...FORM_DEFAULTS.automodel, model: initialModel ?? '' },
      unsloth: {
        ...FORM_DEFAULTS.unsloth,
        model: { ...FORM_DEFAULTS.unsloth.model, name: initialModel ?? '' },
      },
      rl: { ...FORM_DEFAULTS.rl, model: initialModel ?? '' },
    };
  }, [initialModel, initialValues]);

  const form = useForm<CustomizationFormFields>({
    resolver: zodResolver(customizationFormSchema) as unknown as Resolver<CustomizationFormFields>,
    defaultValues,
    mode: 'onChange',
    shouldUnregister: false,
  });

  const backend = useWatch({ control: form.control, name: 'backend' });
  const automodelFinetuningType = useWatch({
    control: form.control,
    name: 'automodel.training.finetuning_type',
  });
  const unslothFinetuningType = useWatch({
    control: form.control,
    name: 'unsloth.training.finetuning_type',
  });
  // Unsloth decides adapter-vs-merged at save time, not via `finetuning_type` — so
  // `producesAdapter` needs this as well. No control binds it today; it is watched
  // rather than read once so the section reacts if one is ever added.
  const unslothSaveMethod = useWatch({
    control: form.control,
    name: 'unsloth.output.save_method',
  });
  // Bound to `grpo.trainingType` rather than `rl.training.type`: the form holds one
  // `rl.training` object, and flipping the union discriminator in place would leave it
  // carrying the other arm's fields. `formToRlCreate` sets `type` from this on submit.
  const grpoTrainingType = useWatch({ control: form.control, name: 'grpo.trainingType' });
  const grpoFinetuningType = useWatch({ control: form.control, name: 'grpo.finetuning_type' });
  const finetuningType = backend === 'automodel' ? automodelFinetuningType : unslothFinetuningType;
  // Gates the LoRA *hyperparameter* controls, so it includes `lora_merged`,
  // which trains with LoRA. It is NOT "the output is an adapter" — a merged run
  // emits full weights. Use `producesAdapter` for anything about serving.
  const usesLoraControls =
    backend !== 'rl' && (finetuningType === 'lora' || finetuningType === 'lora_merged');
  const isDpo = backend === 'rl' && grpoTrainingType !== 'grpo';
  const isGrpo = backend === 'rl' && grpoTrainingType === 'grpo';

  // The serving question, not the training one: only an unmerged LoRA run emits an
  // adapter, and only an adapter is served by a deployment of its *base* model.
  const isAdapterRun = producesAdapter({
    backend,
    automodel: { training: { finetuning_type: automodelFinetuningType } },
    unsloth: {
      training: { finetuning_type: unslothFinetuningType },
      output: { save_method: unslothSaveMethod },
    },
    grpo: { trainingType: grpoTrainingType, finetuning_type: grpoFinetuningType },
  });

  const baseModelRef = useWatch({
    control: form.control,
    name: MODEL_FIELD_BY_BACKEND[backend],
  }) as string | undefined;
  const outputName = useWatch({ control: form.control, name: 'outputName' });

  const readiness = useBaseModelDeploymentReadiness(baseModelRef, { enabled: isAdapterRun });
  // Whether there is a deployment left to create at all. Only the adapter flow can
  // answer "no": its target is the base model, which may already be serving LoRA — or
  // may have failed to resolve at all, which is why this is an allowlist rather than
  // `!== 'serving-lora'`. A full-weight run targets a model that does not exist yet,
  // so nothing can be serving it and nothing had to be looked up — the same reason
  // `launch_model` guards its existing-deployment check with `is_lora`.
  const needsDeployment = isAdapterRun ? CREATE_DEPLOYMENT_STATES.includes(readiness.state) : true;
  const deployRequested = isAdapterRun ? deployBaseModel : deployOutputModel;

  // Separate form: these fields drive their own API calls and are not part of any
  // job payload. Typed exactly `WizardFormValues` so the wizard's field components
  // take its `control` unchanged — see baseDeploymentForm.ts.
  //
  // One form rather than two, because the two sections are mutually exclusive —
  // a run either emits an adapter or it does not. The effect below repoints it.
  const deployForm = useForm<WizardFormValues>({
    resolver: zodResolver(createDeploymentWizardSchema),
    defaultValues: isAdapterRun ? baseDeploymentDefaults() : outputDeploymentDefaults(workspace),
    mode: 'onChange',
  });

  // Keep the nested form pointed at whichever model this run's deployment serves:
  // the base model for an adapter, the run's own output otherwise.
  useEffect(() => {
    const target = isAdapterRun
      ? (baseModelRef ?? '')
      : outputName
        ? `${workspace}/${outputName}`
        : '';
    deployForm.setValue('modelRef', target as WizardFormValues['modelRef']);
    deployForm.setValue('name', baseDeploymentName(isAdapterRun ? baseModelRef : outputName));
    // Re-asserted rather than set once: the adapter flow hides this switch because
    // a LoRA-disabled base refuses the adapter, but the output flow leaves it live,
    // so switching from one to the other could otherwise carry a `false` into a
    // deployment whose whole purpose is to serve the adapter.
    if (isAdapterRun) deployForm.setValue('loraEnabled', true);
  }, [isAdapterRun, baseModelRef, outputName, workspace, deployForm]);

  const { mutateAsync: createAutomodel, isPending: isPendingAutomodel } =
    useCustomizationCreateAutomodelJob({
      mutation: {
        onSuccess: (job) => {
          toast.success('Fine-tuning job started');
          navigate(getWorkspaceCustomizationJobDetailsRoute(workspace, job.name));
        },
        onError: (error: Error) => {
          toast.error(getErrorMessage(error, 'Failed to create fine-tuning job'));
        },
      },
    });

  const { mutateAsync: createUnsloth, isPending: isPendingUnsloth } =
    useCustomizationCreateUnslothJob({
      mutation: {
        onSuccess: (job) => {
          toast.success('Fine-tuning job started');
          navigate(getWorkspaceCustomizationJobDetailsRoute(workspace, job.name));
        },
        onError: (error: Error) => {
          toast.error(getErrorMessage(error, 'Failed to create fine-tuning job'));
        },
      },
    });

  const { mutateAsync: createRl, isPending: isPendingRl } = useCustomizationCreateRlJob({
    mutation: {
      onSuccess: (job) => {
        toast.success('Fine-tuning job started');
        navigate(getWorkspaceCustomizationJobDetailsRoute(workspace, job.name));
      },
      onError: (error: Error) => {
        toast.error(getErrorMessage(error, 'Failed to create fine-tuning job'));
      },
    },
  });

  const isPending = isPendingAutomodel || isPendingUnsloth || isPendingRl;

  /**
   * Anything in flight that the button should report, including the config-creation
   * step that precedes the job mutations. `form.formState.isSubmitting` is true for
   * the whole of `onSubmit`, so it is what actually covers that window.
   */
  const isSubmitting = isPending || form.formState.isSubmitting;

  /**
   * Also blocked while readiness resolves, which is not "busy" but "not yet safe to
   * act on" — the section's `'none'` default is indistinguishable from a real answer
   * until the queries settle. Only for adapter runs, since that is the only case the
   * readiness hook is enabled for.
   */
  const isSubmitBlocked = isSubmitting || (isAdapterRun && readiness.isLoading);

  const onSubmit = async (fields: CustomizationFormFields) => {
    setValidationErrors([]);

    // The config first, and only the config. The job carries its name as
    // `deployment_config` and creates the deployment itself once training finishes
    // — deploying up front reaches the same end state hours earlier and idles a
    // serving GPU for the whole run to get there.
    //
    // Creating it here rather than passing inline parameters is what buys the early
    // failure: `_validate_engine_config` runs synchronously inside
    // create_deployment_config, so a bad engine or missing image surfaces in
    // milliseconds and the job is never submitted. Inline params are validated by
    // the job, after training.
    //
    // Identical for both flows — only the model the config points at differs, and
    // the nested form already carries that. Skipping is allowed: the user may have
    // a serving plan of their own, and the Deployments page can deploy either
    // target at any time afterwards.
    let deploymentConfig: string | undefined;
    if (needsDeployment && deployRequested) {
      const valid = await deployForm.trigger();
      if (!valid) {
        const messages = Object.values(deployForm.formState.errors)
          .map((e) => (e && 'message' in e ? String(e.message) : ''))
          .filter(Boolean);
        setValidationErrors(
          messages.length ? messages : ['Please complete the deployment fields.']
        );
        return;
      }
      const values = deployForm.getValues();
      const configName = configNameFromWizardBaseName(values.name.trim());
      try {
        const { config, reused } = await ensureWorkspaceDeploymentConfig(
          workspace,
          values,
          configName,
          (message) => setDeployStage(message)
        );
        // Adopting an existing config is the intended outcome — one LoRA-enabled
        // deployment of a base serves every adapter trained against it. But it was
        // created by an earlier run and may not match what was just filled in, so
        // say so rather than let the form imply these settings were used. A toast
        // because `onSuccess` navigates away the moment the job is created.
        if (reused) {
          toast.info(
            `Reused the existing deployment configuration "${configName}" ` +
              `(${config.engine}, ${config.executor_config?.gpu ?? '?'} GPU). ` +
              'Its settings take precedence over the ones entered here.'
          );
        }
      } catch (e) {
        setDeployStage(null);
        setValidationErrors([
          getErrorMessage(
            e as Error,
            'Failed to create the deployment configuration. The job was not started.'
          ),
        ]);
        return;
      }
      setDeployStage(null);
      deploymentConfig = configName;
    }

    if (fields.backend === 'automodel') {
      await createAutomodel({
        workspace,
        data: formToAutomodelCreate(fields, deploymentConfig),
      }).catch(() => undefined);
    } else if (fields.backend === 'rl') {
      await createRl({ workspace, data: formToRlCreate(fields, deploymentConfig) }).catch(
        () => undefined
      );
    } else {
      await createUnsloth({
        workspace,
        data: formToUnslothCreate(fields, deploymentConfig),
      }).catch(() => undefined);
    }
  };

  const onInvalid = (formErrors: FieldErrors<CustomizationFormFields>) => {
    const messages = collectErrorMessages(formErrors);
    setValidationErrors(messages.length ? messages : ['Please complete the required fields.']);
  };

  // A seeded config (clone, template, or "start from your own config") is validated the
  // moment it loads, so an imported value that violates the schema — e.g. a numeric of 0
  // against a `.gt(0)` field — surfaces in the banner immediately, not only on submit.
  useEffect(() => {
    if (!initialValues) return;
    const result = customizationFormSchema.safeParse(initialValues);
    if (result.success) return;
    const messages = messagesFromZodIssues(result.error.issues);
    if (messages.length > 0) setValidationErrors(messages);
  }, [initialValues]);

  // Once a banner is showing (seeded config or a rejected submit), keep it in sync as the
  // user corrects fields: revalidate the live values and clear it when they pass. Guarded on
  // an already-shown banner, so it never surfaces errors on an untouched form.
  useEffect(() => {
    const subscription = form.watch(() => {
      setValidationErrors((current) => {
        if (current.length === 0) return current;
        const result = customizationFormSchema.safeParse(form.getValues());
        const next = result.success ? [] : messagesFromZodIssues(result.error.issues);
        // Keep the same array reference when nothing changed, so a keystroke that does not
        // change the error set does not re-render the form.
        return next.length === current.length && next.every((message, i) => message === current[i])
          ? current
          : next;
      });
    });
    return () => subscription.unsubscribe();
  }, [form]);

  useEffect(() => {
    if (validationErrors.length > 0) {
      errorBannerRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [validationErrors]);

  return (
    <AccessibleTitle title="Fine-tune a Model">
      <Stack className="h-full" gap="density-2xl" padding="density-2xl">
        <PageHeader
          slotHeading="Fine-tune a Model"
          slotDescription="Select a model, choose your data, set your parameters and start training in seconds."
        />
        <FormProvider {...form}>
          <form
            className="w-full"
            aria-label="Fine-tune a Model"
            noValidate
            onSubmit={form.handleSubmit(onSubmit, onInvalid)}
          >
            <Stack className="overflow-auto" gap="density-2xl" padding="density-2xl">
              <Flex align="center" justify="center" className="w-full">
                <Panel
                  className="max-w-3xl h-full overflow-auto"
                  elevation="high"
                  density="standard"
                  slotFooter={
                    <Flex className="w-full items-center justify-end gap-2">
                      {deployStage ? (
                        <Text kind="body/regular/sm" color="secondary" className="mr-auto">
                          {deployStage}
                        </Text>
                      ) : null}
                      {/* `isPending` covers only the job mutations, which are the *last*
                          step. Config creation happens before them and takes real time, so
                          on its own `isPending` leaves a window where a second click runs
                          `onSubmit` again and creates a second config.

                          Readiness matters for the opposite reason: while it loads, `state`
                          is still its `'none'` default, which the section reads as "no
                          deployment exists". Submitting then would create a config for a
                          base that may already be serving LoRA. */}
                      <Button
                        type="submit"
                        disabled={isSubmitBlocked}
                        color="brand"
                        aria-busy={isSubmitting}
                      >
                        {isSubmitting ? 'Starting…' : 'Start Fine-Tuning'}
                      </Button>
                    </Flex>
                  }
                >
                  <Stack gap="density-2xl">
                    <BackendSelectionSection />
                    <Divider />
                    <ModelSelectionSection />
                    <Divider />
                    <TrainingMethodSection />
                    {isGrpo && (
                      <>
                        <Divider />
                        <RewardEnvironmentSection />
                      </>
                    )}
                    <Divider />
                    <CustomizationFilesetSelect disabled={isPending} />
                    <Divider />
                    {isGrpo ? <GrpoParametersSection /> : <GeneralParametersSection />}
                    {usesLoraControls && (
                      <>
                        <Divider />
                        <LoraParametersSection />
                      </>
                    )}
                    {isDpo && (
                      <>
                        <Divider />
                        <DpoParametersSection />
                      </>
                    )}
                    <Divider />
                    <IntegrationsSection backend={backend} />
                    <Divider />
                    <ComputeResourcesSection />
                    <Divider />
                    {/* Every run produces something servable, so the section is always
                        offered — what differs is the target. An adapter is served by a
                        deployment of its base model; anything else is served by a
                        deployment of the model the run itself emits. */}
                    {isAdapterRun ? (
                      <DeploymentSection
                        readiness={readiness}
                        control={deployForm.control}
                        errors={deployForm.formState.errors}
                        baseModelRef={baseModelRef ?? ''}
                        deployBaseModel={deployBaseModel}
                        onDeployBaseModelChange={setDeployBaseModel}
                      />
                    ) : (
                      <OutputDeploymentSection
                        control={deployForm.control}
                        errors={deployForm.formState.errors}
                        outputName={outputName ?? ''}
                        deployOutputModel={deployOutputModel}
                        onDeployOutputModelChange={setDeployOutputModel}
                      />
                    )}
                    {validationErrors.length > 0 && (
                      <Banner kind="inline" ref={errorBannerRef} status="error">
                        Please fix the following errors: {validationErrors.join(', ')}
                      </Banner>
                    )}
                  </Stack>
                </Panel>
              </Flex>
            </Stack>
          </form>
        </FormProvider>
      </Stack>
    </AccessibleTitle>
  );
};
