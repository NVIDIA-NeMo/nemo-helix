// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import {
  useAnonymizerCreateRunJob,
  useAnonymizerListEntityLabels,
} from '@nemo/sdk/generated/anonymizer/anonymizer';
import type { PreviewRequest, RunJob } from '@nemo/sdk/generated/anonymizer/schema';
import {
  Banner,
  Button,
  Divider,
  Flex,
  Panel,
  SegmentedControl,
  Stack,
} from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { parseAnonymizerApiError } from '@studio/routes/AnonymizerBuilderRoute/apiErrors';
import { ColumnsSection } from '@studio/routes/AnonymizerBuilderRoute/components/ColumnsSection';
import { DataSourceSection } from '@studio/routes/AnonymizerBuilderRoute/components/DataSourceSection';
import { EntitiesSection } from '@studio/routes/AnonymizerBuilderRoute/components/EntitiesSection';
import { GenerationSection } from '@studio/routes/AnonymizerBuilderRoute/components/GenerationSection';
import { ModelSettingsSection } from '@studio/routes/AnonymizerBuilderRoute/components/ModelSettingsSection';
import { PreviewPanel } from '@studio/routes/AnonymizerBuilderRoute/components/PreviewPanel';
import {
  PANEL_TABS,
  TAB_MODEL_SETTINGS,
  TAB_SOURCE,
} from '@studio/routes/AnonymizerBuilderRoute/constants';
import {
  buildAnonymizerJobRequest,
  buildAnonymizerPreviewRequest,
  type AnonymizerFormData,
} from '@studio/routes/AnonymizerBuilderRoute/schema';
import { useAnonymizerPreview } from '@studio/routes/AnonymizerBuilderRoute/useAnonymizerPreview';
import { useDefaultRoleModels } from '@studio/routes/AnonymizerBuilderRoute/useDefaultRoleModels';
import {
  outputHeadingForStrategy,
  tabForValidationErrors,
} from '@studio/routes/AnonymizerBuilderRoute/utils';
import { getAnonymizerJobRoute, getWorkspaceAnonymizerRoute } from '@studio/routes/utils';
import { useCallback, useState, type FC } from 'react';
import { useFormContext, useWatch, type FieldErrors } from 'react-hook-form';
import { useAuth } from 'react-oidc-context';
import { useNavigate } from 'react-router';

const INCOMPLETE_FORM_MESSAGE = 'Please complete the required fields highlighted below.';

export const AnonymizerBuilderForm: FC = () => {
  const navigate = useNavigate();
  const workspace = useWorkspaceFromPath();
  const { user } = useAuth();
  const form = useFormContext<AnonymizerFormData>();
  const strategy = useWatch({ control: form.control, name: 'strategy' });
  const [activeTab, setActiveTab] = useState<string>(TAB_SOURCE);
  const [submitError, setSubmitError] = useState<string | undefined>(undefined);

  const { isLoading: isLoadingModels } = useDefaultRoleModels();
  const { data: defaultEntityLabels, isLoading: isLoadingEntityLabels } =
    useAnonymizerListEntityLabels(workspace, { query: {} });

  const createJob = useAnonymizerCreateRunJob({
    mutation: {
      onSuccess: (job: RunJob) =>
        navigate(
          job.name
            ? getAnonymizerJobRoute(workspace, job.name)
            : getWorkspaceAnonymizerRoute(workspace)
        ),
      onError: (error) => {
        const { fieldErrors, generalMessages } = parseAnonymizerApiError(error);
        fieldErrors.forEach(({ field, message }) =>
          form.setError(field, { type: 'server', message })
        );
        if (fieldErrors.length) setActiveTab(TAB_SOURCE);
        setSubmitError(
          generalMessages.length
            ? generalMessages.join(' ')
            : fieldErrors.length
              ? undefined
              : getErrorMessage(error, 'Failed to create anonymizer job')
        );
      },
    },
  });

  const showValidationErrors = useCallback((errors: FieldErrors<AnonymizerFormData>) => {
    setActiveTab(tabForValidationErrors(Object.keys(errors)));
    setSubmitError(INCOMPLETE_FORM_MESSAGE);
  }, []);

  // Not `trigger`: `formState` is a proxy tracking what the render body reads, so its `errors`
  // are empty here. Only `handleSubmit` hands back a populated set.
  const getPreviewRequest = useCallback(
    () =>
      new Promise<PreviewRequest | undefined>((resolve) => {
        form
          .handleSubmit(
            (values) => {
              setSubmitError(undefined);
              resolve(buildAnonymizerPreviewRequest(values, defaultEntityLabels?.data ?? []));
            },
            (errors) => {
              showValidationErrors(errors);
              resolve(undefined);
            }
          )()
          .catch(() => {
            setSubmitError(INCOMPLETE_FORM_MESSAGE);
            resolve(undefined);
          });
      }),
    [form, defaultEntityLabels, showValidationErrors]
  );

  const preview = useAnonymizerPreview({
    workspace,
    accessToken: user?.access_token ?? undefined,
    getRequest: getPreviewRequest,
  });

  const onSubmit = form.handleSubmit((values) => {
    setSubmitError(undefined);
    createJob.mutate({
      workspace,
      data: buildAnonymizerJobRequest(values, defaultEntityLabels?.data ?? []),
    });
  }, showValidationErrors);

  const isBusy = createJob.isPending || isLoadingModels || isLoadingEntityLabels;

  return (
    <form className="h-full" noValidate onSubmit={onSubmit}>
      <Flex className="h-full" gap="0">
        <Panel
          className="w-[400px] h-full"
          elevation="high"
          density="standard"
          attributes={{ PanelContent: { className: 'flex-1 min-h-0 overflow-auto' } }}
        >
          <Stack gap="density-2xl">
            <Flex align="center" gap="density-md">
              <SegmentedControl
                className="flex-1"
                value={activeTab}
                onValueChange={setActiveTab}
                items={PANEL_TABS}
              />
              {preview.isPreviewing ? (
                <Button kind="secondary" onClick={preview.stopPreview} type="button">
                  Stop
                </Button>
              ) : (
                <Button
                  color="brand"
                  disabled={isBusy}
                  kind="primary"
                  onClick={() => void preview.runPreview()}
                  type="button"
                >
                  Preview
                </Button>
              )}
            </Flex>

            {submitError ? (
              <Banner kind="inline" status="error">
                {submitError}
              </Banner>
            ) : null}

            <div className={activeTab === TAB_SOURCE ? undefined : 'hidden'}>
              <Stack gap="density-2xl">
                <DataSourceSection />
                <Divider orientation="horizontal" width="small" />
                <GenerationSection />
                <Divider orientation="horizontal" width="small" />
                <ColumnsSection />
                <Divider orientation="horizontal" width="small" />
                <EntitiesSection />
              </Stack>
            </div>
            <div className={activeTab === TAB_MODEL_SETTINGS ? undefined : 'hidden'}>
              <ModelSettingsSection />
            </div>
          </Stack>
        </Panel>

        <PreviewPanel
          pendingOutputHeading={outputHeadingForStrategy(strategy)}
          preview={preview}
          slotActions={
            <Button color="brand" disabled={isBusy} kind="primary" type="submit">
              Full Run
            </Button>
          }
        />
      </Flex>
    </form>
  );
};
