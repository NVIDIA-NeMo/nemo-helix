// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { useDataDesignerCreateJob } from '@nemo/sdk/generated/data-designer/data-designer';
import type { CreateJobRequest as DataDesignerJobRequest } from '@nemo/sdk/generated/data-designer/schema';
import { useModelsListProviders } from '@nemo/sdk/generated/platform/model-providers';
import { Button, CodeSnippet, Flex, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import { JobBasics } from '@studio/components/NewDataDesignerJobForm/JobBasics';
import { JobRequestGenerator } from '@studio/components/NewDataDesignerJobForm/JobRequestGenerator';
import { formatPreviewLogsForDisplay } from '@studio/components/NewDataDesignerJobForm/previewApi';
import { usePreview } from '@studio/components/NewDataDesignerJobForm/usePreview';
import {
  type DataDesignerModelOption,
  getCloneJobRequestFromState,
  modelsFromProviders,
  parseJsonContentToJobRequest,
  sanitizeJobRequestName,
} from '@studio/components/NewDataDesignerJobForm/utils';
import { DEFAULT_BUILD_MODEL_NAME, DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useOidcBearerToken } from '@studio/providers/auth/useOidcBearerToken';
import {
  getDataDesignerJobDetailsRoute,
  getDataDesignerJobListRoute,
  getWorkspaceInferenceProvidersRoute,
} from '@studio/routes/utils';
import { type FC, useCallback, useEffect, useMemo } from 'react';
import { useForm, useWatch } from 'react-hook-form';
import { Link, useLocation, useNavigate } from 'react-router';
import { z } from 'zod';

export type { DataDesignerModelOption };

const sharedFields = {
  name: z
    .string()
    .refine(
      (val) => !val || !/\s/.test(val),
      'Name must not contain spaces. Use hyphens or underscores (e.g. my-data-job).'
    ),
  jobDescription: z.string(),
  modelRef: z.string().min(1, 'Please select a model'),
  rows: z.number({ required_error: 'Rows is required' }).min(1, 'Must be at least 1'),
  inferenceSecret: z.string().optional(),
  // Editable job request JSON. Must contain a valid config before submit.
  jsonContent: z
    .string()
    .refine(
      (value) => !!parseJsonContentToJobRequest(value).jobRequest?.spec?.config,
      'Generate or provide valid job JSON before creating.'
    ),
};

const newDataDesignerJobFormSchema = z.object({
  ...sharedFields,
  description: z
    .string()
    .min(1, 'Description is required')
    .min(
      10,
      'Please provide at least a short description (e.g. "100 rows of product reviews with sentiment")'
    ),
});

// When cloning, the JSON config is pre-filled, so the natural-language description used for
// generation is optional — the user can submit the cloned config without describing it again.
const cloneDataDesignerJobFormSchema = z.object({
  ...sharedFields,
  description: z.string(),
});

export type NewDataDesignerJobFormFields = z.infer<typeof newDataDesignerJobFormSchema>;

export const NewDataDesignerJobForm: FC = () => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const { state } = useLocation();
  const bearerToken = useOidcBearerToken();

  const modelRef = `${workspace}/${DEFAULT_BUILD_MODEL_NAME}`;

  const clonedJobRequest = useMemo(() => getCloneJobRequestFromState(state), [state]);
  const isClone = clonedJobRequest != null;
  const initialJsonContent = useMemo(
    () => (clonedJobRequest ? JSON.stringify(clonedJobRequest, null, 2) : ''),
    [clonedJobRequest]
  );

  const {
    control,
    handleSubmit,
    setValue,
    getValues,
    formState: { errors },
  } = useForm<NewDataDesignerJobFormFields>({
    resolver: zodResolver(isClone ? cloneDataDesignerJobFormSchema : newDataDesignerJobFormSchema),
    defaultValues: {
      name: clonedJobRequest?.name ?? '',
      jobDescription: clonedJobRequest?.description ?? '',
      description: '',
      modelRef,
      rows: clonedJobRequest?.spec?.num_records ?? 100,
      jsonContent: initialJsonContent,
    },
  });

  const { data: providersPage, isLoading: isLoadingModels } = useModelsListProviders(
    workspace,
    { page_size: DEFAULT_LARGE_PAGE_SIZE },
    { query: {} }
  );
  const models = useMemo(
    () => modelsFromProviders(providersPage?.data ?? []),
    [providersPage?.data]
  );
  const selectedModel = models?.find((m) => m.id === modelRef);
  const modelNotFound = !isLoadingModels && !selectedModel;

  const setJsonContent = useCallback(
    (value: string) => setValue('jsonContent', value, { shouldDirty: true }),
    [setValue]
  );

  const watchedJsonContent = useWatch({ control, name: 'jsonContent' });
  useEffect(() => {
    const numRecords = parseJsonContentToJobRequest(watchedJsonContent ?? '').jobRequest?.spec
      ?.num_records;
    if (numRecords != null) {
      setValue('rows', numRecords);
    }
  }, [watchedJsonContent, setValue]);

  const getCurrentConfig = useCallback(
    () => parseJsonContentToJobRequest(getValues('jsonContent')).jobRequest?.spec?.config,
    [getValues]
  );
  const { previewLogs, isPreviewing, runPreview } = usePreview({
    workspace,
    accessToken: bearerToken,
    getCurrentConfig,
  });

  const createJob = useDataDesignerCreateJob();
  const isSubmitting = createJob.isPending;
  const submitError = createJob.error instanceof Error ? createJob.error.message : null;

  const onSubmit = useCallback(
    async (fields: NewDataDesignerJobFormFields) => {
      const current = parseJsonContentToJobRequest(fields.jsonContent).jobRequest;
      if (!current?.spec?.config) return;

      const fromSpec = current.spec.num_records;
      const fromForm = Number(fields.rows);
      const numRecords = fromForm || fromSpec || 10;
      const merged: DataDesignerJobRequest = {
        ...current,
        spec: { ...current.spec, num_records: numRecords },
      };
      if (fields.name.trim()) merged.name = fields.name.trim();
      if (fields.jobDescription.trim()) merged.description = fields.jobDescription.trim();
      const toSubmit = sanitizeJobRequestName(merged);

      try {
        const created = await createJob.mutateAsync({ workspace, data: toSubmit });
        if (created?.name) {
          navigate(getDataDesignerJobDetailsRoute(workspace, created.name));
        } else {
          navigate(getDataDesignerJobListRoute(workspace));
        }
      } catch {
        // Error surfaced via createJob.error
      }
    },
    [workspace, createJob, navigate]
  );

  if (modelNotFound) {
    return (
      <Panel elevation="high" density="standard">
        <ErrorMessage
          header="Model Not Available"
          message={
            <>
              Add the{' '}
              <Link
                to={getWorkspaceInferenceProvidersRoute(workspace, { preset: 'build' })}
                className="underline"
              >
                NVIDIA Build Inference Provider
              </Link>{' '}
              to your Workspace to enable this feature.
            </>
          }
          slotFooter={
            <Button
              kind="secondary"
              onClick={() =>
                navigate(getWorkspaceInferenceProvidersRoute(workspace, { preset: 'build' }))
              }
            >
              Go to Inference Providers
            </Button>
          }
          height="auto"
        />
      </Panel>
    );
  }

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <Stack gap="density-2xl">
        <JobBasics
          control={control}
          nameName="name"
          rowsName="rows"
          descriptionName="jobDescription"
          disabled={isPreviewing || isSubmitting}
        />

        <Panel
          elevation="high"
          density="standard"
          slotFooter={
            <Flex className="w-full justify-between">
              <Button
                type="button"
                kind="secondary"
                onClick={() => navigate(getDataDesignerJobListRoute(workspace))}
                disabled={isSubmitting}
              >
                Cancel
              </Button>
              <Flex gap="density-md">
                <LoadingButton
                  type="button"
                  kind="secondary"
                  onClick={runPreview}
                  loading={isPreviewing}
                >
                  Preview
                </LoadingButton>
                <LoadingButton type="submit" color="brand" loading={isSubmitting}>
                  Create Job
                </LoadingButton>
              </Flex>
            </Flex>
          }
        >
          {/* Data Specification */}
          <Stack gap="density-lg">
            <Stack gap="density-xs">
              <Text kind="label/bold/lg">Data Specification</Text>
              <Text kind="body/regular/sm">
                Describe the type of data you want to generate. The selected model will convert this
                into a job specification, which you can review and edit before submitting.
              </Text>
            </Stack>
            <JobRequestGenerator
              workspace={workspace}
              modelRef={modelRef}
              provider={selectedModel?.model_providers?.[0] ?? ''}
              servedModelName={selectedModel?.served_model_name ?? ''}
              control={control}
              descriptionName="description"
              jsonContentName="jsonContent"
              setJsonContent={setJsonContent}
              disabled={isPreviewing || isSubmitting}
            />
          </Stack>
        </Panel>

        {errors.jsonContent && (
          <Text kind="body/regular/sm" className="text-danger">
            {errors.jsonContent.message}
          </Text>
        )}

        {submitError && (
          <Text kind="body/regular/sm" className="text-danger">
            {submitError}
          </Text>
        )}

        <CodeSnippet
          value={
            previewLogs ? formatPreviewLogsForDisplay(previewLogs) : 'Run Preview to see logs.'
          }
          language="json"
          kind="block"
          attributes={{ CodeSnippetCode: { className: 'max-h-[600px]' } }}
        />
      </Stack>
    </form>
  );
};
