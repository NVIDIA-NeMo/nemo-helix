// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import {
  checkDatasetQuality,
  type DatasetQualityReport,
} from '@nemo/common/src/utils/datasetQuality';
import { handleFormErrorsGeneric } from '@nemo/common/src/utils/forms/error';
import { type FilesetOutput, FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import { Button, SidePanel, Stack } from '@nvidia/foundations-react-core';
import type { NameCheckStatus } from '@studio/components/form/entityNameCheck';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { CustomFilesetForm } from '@studio/routes/FilesetNewRoute/CustomFilesetForm';
import {
  DatasetCreateFilesetFormSchema,
  DatasetFormFields,
} from '@studio/routes/FilesetNewRoute/types';
import { useCreateFileset } from '@studio/routes/FilesetNewRoute/useCreateFileset';
import { CreateSecretModal } from '@studio/routes/SecretsListRoute/CreateSecretModal';
import { getWorkspaceFilesetsRoute } from '@studio/routes/utils';
import { isHuggingFaceUrl, isNgcUrl } from '@studio/util/storageConfigFromUrl';
import { FC, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useForm } from 'react-hook-form';
import { useNavigate } from 'react-router';

export const FilesetNewRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const [isSubmitPending, setIsSubmitPending] = useState(false);
  // Once created, the fileset is in the list the name check reads, and must not count against
  // itself while files upload or after an upload fails. A retry after a failed upload reuses it.
  const [createdFileset, setCreatedFileset] = useState<FilesetOutput>();
  // Reported by the name field, so typing does not re-render the whole panel.
  const [nameStatus, setNameStatus] = useState<NameCheckStatus>();
  const nameTaken = nameStatus === 'conflict';
  const [qualityReports, setQualityReports] = useState<DatasetQualityReport[]>([]);
  const [isValidating, setIsValidating] = useState(false);
  const qualityReportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (qualityReports.length > 0) {
      qualityReportRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [qualityReports]);

  const navigate = useNavigate();

  const [storageTab, setStorageTab] = useState<'local' | 'external'>('local');
  const [createSecretModalOpen, setCreateSecretModalOpen] = useState(false);

  const {
    control,
    handleSubmit,
    setValue,
    getValues,
    watch,
    setError,
    formState: { errors },
  } = useForm<DatasetFormFields>({
    resolver: zodResolver(DatasetCreateFilesetFormSchema),
    defaultValues: {
      name: '',
      description: '',
      files: undefined,
      url: '',
      secretKey: '',
      // Default chosen for least surprise: the page previously hardcoded purpose=dataset.
      purpose: FilesetPurpose.dataset,
    },
    mode: 'onChange',
  });

  const url = watch('url');
  const purpose = watch('purpose');

  useEffect(() => {
    if (purpose !== FilesetPurpose.dataset) {
      setQualityReports([]);
      setIsValidating(false);
    }
  }, [purpose]);

  const selectedSecretName = watch('secretKey');
  const secretKeyLabel = useMemo(() => {
    if (!url?.trim()) return 'Secret Key';
    try {
      if (isHuggingFaceUrl(new URL(url))) return 'HuggingFace Token (optional)';
      if (isNgcUrl(new URL(url))) return 'NGC API Key';
    } catch {
      // invalid URL
    }
    return 'Secret Key';
  }, [url]);

  /**
   * Runs dataset quality checks on newly selected JSONL files and updates the report state.
   * Only runs when purpose is 'dataset'; clears reports for other purposes or non-JSONL files.
   */
  const handleFilesChange = useCallback(
    async (files: File[]) => {
      setValue('files', files, { shouldValidate: false });

      if (purpose !== FilesetPurpose.dataset) {
        setQualityReports([]);
        return;
      }

      const jsonlFiles = files.filter((f) => f.name.endsWith('.jsonl'));
      if (jsonlFiles.length === 0) {
        setQualityReports([]);
        return;
      }

      setIsValidating(true);
      const reports = await Promise.all(jsonlFiles.map(checkDatasetQuality));
      setQualityReports(reports);
      setIsValidating(false);
    },
    [purpose, setValue]
  );

  const hasValidationErrors =
    purpose === FilesetPurpose.dataset && qualityReports.some((r) => r.hasErrors);

  const onSubmit = useCreateFileset({
    workspace,
    storageTab,
    hasValidationErrors,
    nameTaken,
    getValues,
    setError,
    setIsSubmitPending,
    createdFileset,
    onCreated: setCreatedFileset,
  });

  const handleClose = useCallback(() => {
    navigate(getWorkspaceFilesetsRoute(workspace), { flushSync: true });
  }, [navigate, workspace]);

  const handleSecretCreated = useCallback(
    (secretName: string) => {
      setValue('secretKey', secretName, { shouldValidate: true });
    },
    [setValue]
  );

  return (
    <>
      <SidePanel
        slotHeading="Create Fileset"
        side="right"
        open
        onOpenChange={handleClose}
        className="max-w-[600px] w-full"
        bordered
        modal
        slotFooter={
          <>
            <Button kind="tertiary" onClick={handleClose}>
              Cancel
            </Button>
            <Button
              type="button"
              color="brand"
              disabled={isSubmitPending || hasValidationErrors || nameTaken}
              onClick={handleSubmit(
                onSubmit,
                handleFormErrorsGeneric({ title: 'Fileset New Form Errors' })
              )}
            >
              {isSubmitPending ? 'Creating Fileset...' : 'Create Fileset'}
            </Button>
          </>
        }
      >
        <Stack className="h-full overflow-auto w-full" gap="density-xl">
          <CustomFilesetForm
            control={control}
            errors={errors}
            setValue={setValue}
            isSubmitPending={isSubmitPending}
            purpose={purpose}
            workspace={workspace}
            ownName={createdFileset?.name}
            onNameStatusChange={setNameStatus}
            storageTab={storageTab}
            setStorageTab={setStorageTab}
            selectedSecretName={selectedSecretName}
            secretKeyLabel={secretKeyLabel}
            isValidating={isValidating}
            qualityReports={qualityReports}
            qualityReportRef={qualityReportRef}
            onFormSubmit={handleSubmit(
              onSubmit,
              handleFormErrorsGeneric({ title: 'Fileset New Form Errors' })
            )}
            onFilesChange={handleFilesChange}
            onClearQualityReports={() => setQualityReports([])}
            onRequestNewSecret={() => setCreateSecretModalOpen(true)}
          />
        </Stack>
      </SidePanel>
      <CreateSecretModal
        workspace={workspace}
        open={createSecretModalOpen}
        onClose={() => setCreateSecretModalOpen(false)}
        onSecretCreated={handleSecretCreated}
      />
    </>
  );
};
