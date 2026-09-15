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
  DEFAULT_DEPLOY_BASE_MODEL,
} from '@studio/components/NewCustomizationForm/baseDeploymentForm';
import { ComputeResourcesSection } from '@studio/components/NewCustomizationForm/ComputeResourcesSection';
import { DeploymentSection } from '@studio/components/NewCustomizationForm/DeploymentSection';
import { DpoParametersSection } from '@studio/components/NewCustomizationForm/DpoParametersSection';
import { GeneralParametersSection } from '@studio/components/NewCustomizationForm/GeneralParametersSection';
import { GrpoParametersSection } from '@studio/components/NewCustomizationForm/GrpoParametersSection';
import { IntegrationsSection } from '@studio/components/NewCustomizationForm/IntegrationsSection';
import { LoraParametersSection } from '@studio/components/NewCustomizationForm/LoraParametersSection';
import { ModelSelectionSection } from '@studio/components/NewCustomizationForm/ModelSelectionSection';
import { RewardEnvironmentSection } from '@studio/components/NewCustomizationForm/RewardEnvironmentSection';
import { TrainingMethodSection } from '@studio/components/NewCustomizationForm/TrainingMethodSection';
import { useBaseModelDeploymentReadiness } from '@studio/hooks/useBaseModelDeploymentReadiness';
import {
  configNameFromWizardBaseName,
  createDeploymentWizardSchema,
  type WizardFormValues,
} from '@studio/routes/NewDeploymentRoute/schema';
import { createWorkspaceDeploymentConfig } from '@studio/routes/NewDeploymentRoute/useCreateDeploymentBySource';
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
  const [deployBaseModel, setDeployBaseModel] = useState(DEFAULT_DEPLOY_BASE_MODEL);

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

  const readiness = useBaseModelDeploymentReadiness(baseModelRef, { enabled: isAdapterRun });
  const needsBaseDeployment = isAdapterRun && readiness.state !== 'serving-lora';

  // Separate form: these fields drive their own API calls and are not part of any
  // job payload. Typed exactly `WizardFormValues` so the wizard's field components
  // take its `control` unchanged — see baseDeploymentForm.ts.
  const deployForm = useForm<WizardFormValues>({
    resolver: zodResolver(createDeploymentWizardSchema),
    defaultValues: baseDeploymentDefaults(),
    mode: 'onChange',
  });

  // Keep the nested form pointed at whatever base model is currently picked.
  useEffect(() => {
    deployForm.setValue('modelRef', (baseModelRef ?? '') as WizardFormValues['modelRef']);
    deployForm.setValue('name', baseDeploymentName(baseModelRef));
  }, [baseModelRef, deployForm]);

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
    // Skipping is allowed: the user may already have a serving plan of their own.
    // The section warns that the adapter will not be servable until the base model
    // is deployed, and the Deployments page can do that at any time afterwards.
    let deploymentConfig: string | undefined;
    if (needsBaseDeployment && deployBaseModel) {
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
        await createWorkspaceDeploymentConfig(workspace, values, configName, (message) =>
          setDeployStage(message)
        );
      } catch (e) {
        setDeployStage(null);
        setValidationErrors([
          getErrorMessage(
            e as Error,
            'Failed to create the base model deployment configuration. The job was not started.'
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
    const messages: string[] = [];
    const collect = (node: unknown) => {
      if (!node || typeof node !== 'object') return;
      if ('message' in node && typeof (node as { message?: unknown }).message === 'string') {
        messages.push((node as { message: string }).message);
        return;
      }
      Object.values(node as Record<string, unknown>).forEach(collect);
    };
    collect(formErrors);
    setValidationErrors(
      messages.length ? Array.from(new Set(messages)) : ['Please complete the required fields.']
    );
  };

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
                      <Button type="submit" disabled={isPending} color="brand">
                        {isPending ? 'Starting…' : 'Start Fine-Tuning'}
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
                    {isAdapterRun && (
                      <>
                        <Divider />
                        <DeploymentSection
                          readiness={readiness}
                          control={deployForm.control}
                          errors={deployForm.formState.errors}
                          baseModelRef={baseModelRef ?? ''}
                          deployBaseModel={deployBaseModel}
                          onDeployBaseModelChange={setDeployBaseModel}
                        />
                      </>
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
