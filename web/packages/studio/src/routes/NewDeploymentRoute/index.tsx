/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { zodResolver } from '@hookform/resolvers/zod';
import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { resourceRefSchema, type ResourceRef } from '@nemo/common/src/types';
import { generateDefaultName } from '@nemo/common/src/utils/generateDefaultName';
import {
  Button,
  Flex,
  PageHeader,
  Panel,
  SegmentedControl,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { QUERY_PARAMETERS } from '@studio/routes/constants';
import { AdvancedSettingsAccordion } from '@studio/routes/NewDeploymentRoute/AdvancedSettingsAccordion';
import { HuggingFaceSourceFields } from '@studio/routes/NewDeploymentRoute/HuggingFaceSourceFields';
import { NgcSourceFields } from '@studio/routes/NewDeploymentRoute/NgcSourceFields';
import {
  createDeploymentWizardSchema,
  defaultWizardValues,
  WORKSPACE_PICKER_FILESET,
  WORKSPACE_PICKER_MODEL,
  SOURCE_HF,
  SOURCE_WORKSPACE,
  SOURCE_NGC,
  type WizardFormValues,
} from '@studio/routes/NewDeploymentRoute/schema';
import { useCreateDeploymentBySource } from '@studio/routes/NewDeploymentRoute/useCreateDeploymentBySource';
import { useHuggingFaceNameDefault } from '@studio/routes/NewDeploymentRoute/useHuggingFaceNameDefault';
import { WorkspaceSourceFields } from '@studio/routes/NewDeploymentRoute/WorkspaceSourceFields';
import { getWorkspaceDeploymentsRoute } from '@studio/routes/utils';
import { FC, useCallback, useMemo, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useNavigate, useSearchParams } from 'react-router';

/** Deep-link prefill: seeds the Workspace source with the given ref. */
export interface CreateDeploymentPrefill {
  /** `<workspace>/<name>` reference to a model entity. Takes precedence over `fileset`. */
  modelRef?: ResourceRef;
  /** `<workspace>/<name>` reference to a fileset. */
  fileset?: ResourceRef;
}

function getResourceRefSearchParam(params: URLSearchParams, name: string): ResourceRef | undefined {
  const value = params.get(name);
  if (!value) return undefined;
  const parsed = resourceRefSchema.safeParse(value);
  return parsed.success ? parsed.data : undefined;
}

function valuesFromPrefill(prefill: CreateDeploymentPrefill | undefined): WizardFormValues {
  const base = defaultWizardValues();
  if (!prefill) return base;
  if (prefill.modelRef) {
    return {
      ...base,
      source: SOURCE_WORKSPACE,
      workspacePickerType: WORKSPACE_PICKER_MODEL,
      modelRef: prefill.modelRef,
    };
  }
  if (prefill.fileset) {
    return {
      ...base,
      source: SOURCE_WORKSPACE,
      workspacePickerType: WORKSPACE_PICKER_FILESET,
      fileset: prefill.fileset,
    };
  }
  return base;
}

export const NewDeploymentRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const searchParamString = searchParams.toString();

  useBreadcrumbs({
    items: [
      { slotLabel: 'Deployments', href: getWorkspaceDeploymentsRoute(workspace) },
      { slotLabel: 'Create Deployment' },
    ],
  });

  const { createDeploymentFromWizard, isSubmitting, submitError, statusMessage } =
    useCreateDeploymentBySource(workspace);
  const [advancedAccordion, setAdvancedAccordion] = useState<string>();

  // Read once at mount: the page owns the form after that, and re-seeding on a
  // URL change would silently discard whatever the user has typed.
  const initialValues = useMemo(() => {
    const params = new URLSearchParams(searchParamString);
    const modelRef = getResourceRefSearchParam(params, QUERY_PARAMETERS.model);
    if (modelRef) return valuesFromPrefill({ modelRef });
    const fileset = getResourceRefSearchParam(params, QUERY_PARAMETERS.fileset);
    return valuesFromPrefill(fileset ? { fileset } : undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const {
    control,
    handleSubmit,
    reset,
    getValues,
    setValue,
    watch,
    formState: { errors },
  } = useForm<WizardFormValues>({
    resolver: zodResolver(createDeploymentWizardSchema),
    defaultValues: initialValues,
    mode: 'onChange',
    disabled: isSubmitting,
  });

  const source = watch('source');

  // On the HuggingFace source, default the name from the repo id until the user
  // types their own.
  useHuggingFaceNameDefault(control, setValue);

  const goToList = useCallback(() => {
    navigate(getWorkspaceDeploymentsRoute(workspace));
  }, [navigate, workspace]);

  const onSubmit: SubmitHandler<WizardFormValues> = useCallback(
    (values) => createDeploymentFromWizard(values, goToList),
    [createDeploymentFromWizard, goToList]
  );

  return (
    <AccessibleTitle title="Create Deployment">
      <Stack className="h-full" gap="density-2xl" padding="density-2xl">
        <PageHeader
          slotHeading="Create Deployment"
          slotDescription="Pick where the model comes from, choose how it is served, and start serving it in minutes."
        />
        <form
          className="w-full"
          aria-label="Create Deployment"
          onSubmit={handleSubmit(onSubmit)}
          noValidate
        >
          <Stack className="overflow-auto" gap="density-2xl" padding="density-2xl">
            <Flex align="center" justify="center" className="w-full">
              <Panel
                className="h-full w-full max-w-[600px] overflow-auto"
                elevation="high"
                density="standard"
                slotFooter={
                  <Stack gap="2">
                    {statusMessage ? (
                      <Text kind="body/regular/sm" color="secondary" className="mr-auto max-w-full">
                        {statusMessage}
                      </Text>
                    ) : null}
                    {submitError ? (
                      <Text kind="body/regular/sm" className="mr-auto max-w-full text-red-400">
                        {submitError}
                      </Text>
                    ) : null}
                    <Flex justify="end" gap="density-lg" className="w-full flex-wrap">
                      <Button
                        kind="tertiary"
                        type="button"
                        disabled={isSubmitting}
                        onClick={() => {
                          if (!isSubmitting) goToList();
                        }}
                      >
                        Cancel
                      </Button>
                      <LoadingButton type="submit" loading={isSubmitting} disabled={isSubmitting}>
                        Deploy
                      </LoadingButton>
                    </Flex>
                  </Stack>
                }
              >
                <Stack gap="density-xl">
                  <SegmentedControl
                    className="w-full"
                    value={source}
                    onValueChange={(v) => {
                      const keepName = getValues('name') || generateDefaultName();
                      reset({
                        ...defaultWizardValues(),
                        source: v as typeof SOURCE_NGC | typeof SOURCE_HF | typeof SOURCE_WORKSPACE,
                        name: keepName,
                      });
                    }}
                    items={[
                      { value: SOURCE_HF, children: 'HuggingFace' },
                      { value: SOURCE_WORKSPACE, children: 'Workspace' },
                      { value: SOURCE_NGC, children: 'NGC NIM Container' },
                    ]}
                  />
                  {(source === SOURCE_HF || source === SOURCE_WORKSPACE) && (
                    <Text kind="body/regular/md">
                      Choose the engine that serves this model. vLLM runs any supported architecture
                      from a default image; NIM needs an image built for the specific architecture.
                    </Text>
                  )}

                  <ControlledTextInput
                    useControllerProps={{ control, name: 'name' }}
                    name="name"
                    label="Name"
                    formFieldProps={{
                      slotInfo: 'Name used across all related assets.',
                      slotError: errors.name?.message,
                    }}
                  />

                  {source === SOURCE_NGC && <NgcSourceFields control={control} errors={errors} />}
                  {source === SOURCE_HF && (
                    <HuggingFaceSourceFields
                      control={control}
                      errors={errors}
                      queryEnabled={!!workspace}
                      workspace={workspace}
                    />
                  )}
                  {source === SOURCE_WORKSPACE && (
                    <WorkspaceSourceFields
                      control={control}
                      errors={errors}
                      queryEnabled={!!workspace}
                      workspace={workspace}
                      onPickerTypeChange={(value) => {
                        setValue('workspacePickerType', value, {
                          shouldDirty: true,
                          shouldTouch: true,
                          shouldValidate: true,
                        });
                        if (value === WORKSPACE_PICKER_MODEL) {
                          setValue('fileset', '', { shouldDirty: true });
                        }
                        if (value === WORKSPACE_PICKER_FILESET) {
                          setValue('modelRef', '', { shouldDirty: true });
                        }
                      }}
                    />
                  )}

                  <AdvancedSettingsAccordion
                    advancedAccordion={advancedAccordion}
                    control={control}
                    errors={errors}
                    onAdvancedAccordionChange={setAdvancedAccordion}
                  />
                </Stack>
              </Panel>
            </Flex>
          </Stack>
        </form>
      </Stack>
    </AccessibleTitle>
  );
};
